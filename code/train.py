import argparse
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader.dsa_dataset import DSASequenceDataset
from models import MotionUNet
from utils.common import clean_state_dict, init_distributed, reduce_mean, set_seed, setup_logging, unwrap
from utils.losses import MotionLoss, PhysicsLoss, SegmentationLoss
from utils.lr_scheduler import IterationScheduler
from utils.metrics import batch_metrics

GRAD_CLIP = 12.0
LOSS_KEYS = ('loss', 'ce', 'dice', 'reg', 'sim', 'smooth', 'filling')


def parse_args():
    p = argparse.ArgumentParser(description='Train MACOS')
    p.add_argument('--data_dir', type=str, default='./data/DSA_sequences', help='dataset directory (contains data_splits.json)')
    p.add_argument('--val_split', type=str, default='test', help='split evaluated after every epoch and used to pick best_model.pth')
    p.add_argument('--output_dir', type=str, default='./results', help='root directory for results')
    p.add_argument('--tag', type=str, default='', help='suffix appended to the run name')
    p.add_argument('--frame_num', type=int, default=6, help='frames sampled per sequence')
    p.add_argument('--deep_supervision', action='store_true', help='multi-scale supervision of the decoder')
    p.add_argument('--augmentation', action='store_true', help='data augmentation on the training split')
    p.add_argument('--batch_size', type=int, default=4, help='batch size per GPU')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.99, help='SGD momentum')
    p.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'poly'])
    p.add_argument('--warmup_epochs', type=int, default=30)
    p.add_argument('--min_lr_ratio', type=float, default=5e-5)
    p.add_argument('--poly_power', type=float, default=0.9)
    p.add_argument('--ce_weight', type=float, default=1.0)
    p.add_argument('--dice_weight', type=float, default=1.0)
    p.add_argument('--focal_weight', type=float, default=0.8, help='weight of the focal term (0 disables it)')
    p.add_argument('--focal_alpha', type=float, default=0.25)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--reg_weight', type=float, default=0.3, help='lambda_1, Global Motion Loss weight')
    p.add_argument('--reg_type', type=str, default='mask', choices=['mask', 'normal'],
                   help='mask: photometric term on predicted background only; normal: on the whole frame')
    p.add_argument('--filling_weight', type=float, default=0.5, help='lambda_2, Physics-Informed Vessel Loss weight')
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--gpu_id', type=int, default=0, help='GPU used for single-GPU training')
    p.add_argument('--resume', type=str, default=None, help='checkpoint to resume from (model, optimizer, schedule)')
    p.add_argument('--init_weights', type=str, default=None, help='checkpoint whose model weights initialise the network')
    return p.parse_args()


def key_frame_logits(outputs):
    """Per-frame logits (tensor or list with deep supervision) -> key-frame logits in the same structure."""
    if isinstance(outputs, list):
        return [o[:, -1] for o in outputs]
    return outputs[:, -1]


def train_epoch(model, loader, seg_loss, motion_loss, physics_loss, optimizer, scheduler, scaler, device, epoch, args, logger, rank, world_size):
    model.train()
    sums = {k: 0.0 for k in LOSS_KEYS}
    metrics = []
    bar = tqdm(loader, desc=f'Training epoch {epoch}') if rank == 0 else loader
    for batch in bar:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'][:, -1].to(device, non_blocking=True)
        B, T = images.shape[:2]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(images)
            predictions = outputs['predictions']
            key_logits = key_frame_logits(predictions)
            loss, ce, dice = seg_loss(key_logits, targets)
            reg = sim = smooth = filling = torch.zeros((), device=device)
            if outputs['displacement_field'] is not None:
                flow = outputs['displacement_field']
                main = predictions[0] if isinstance(predictions, list) else predictions
                reg, sim, smooth = motion_loss(outputs['warped_image'], outputs['fixed_image'], flow,
                                               main if args.reg_type == 'mask' else None)
                filling = physics_loss(main, flow.reshape(B, T - 1, 2, *flow.shape[-2:]))
                loss = loss + args.reg_weight * reg + args.filling_weight * filling
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        lr = scheduler.step()
        for k, v in zip(LOSS_KEYS, (loss, ce, dice, reg, sim, smooth, filling)):
            sums[k] += v.item()
        with torch.no_grad():
            metrics.append(batch_metrics(key_logits[0] if isinstance(key_logits, list) else key_logits, targets))
        if rank == 0:
            bar.set_postfix(loss=f'{loss.item():.4f}', sim=f'{sim.item():.4f}', filling=f'{filling.item():.4f}',
                            dice=f'{metrics[-1]["dice"]:.4f}', lr=f'{lr:.2e}')
    avg = {k: reduce_mean(v / len(loader), world_size, device) for k, v in sums.items()}
    avg_metrics = {k: reduce_mean(float(np.mean([m[k] for m in metrics])), world_size, device) for k in metrics[0]}
    if rank == 0:
        logger.info(f'Training epoch {epoch}: loss={avg["loss"]:.4f}, ce={avg["ce"]:.4f}, dice_loss={avg["dice"]:.4f}, '
                    f'motion={avg["reg"]:.4f} (sim={avg["sim"]:.4f}, smooth={avg["smooth"]:.4f}), physics={avg["filling"]:.4f}')
        logger.info(f'Training metrics: dice={avg_metrics["dice"]:.4f}, iou={avg_metrics["iou"]:.4f}, acc={avg_metrics["accuracy"]:.4f}')
    return avg['loss'], avg_metrics


@torch.no_grad()
def validate_epoch(model, loader, device, epoch, logger, rank, world_size):
    model.eval()
    metrics = []
    bar = tqdm(loader, desc=f'Validation epoch {epoch}') if rank == 0 else loader
    for batch in bar:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'][:, -1].to(device, non_blocking=True)
        logits = model(images, key_frame_only=True)['predictions']
        metrics.append(batch_metrics(logits, targets))
        if rank == 0:
            bar.set_postfix(dice=f'{metrics[-1]["dice"]:.4f}', iou=f'{metrics[-1]["iou"]:.4f}')
    avg_metrics = {k: reduce_mean(float(np.mean([m[k] for m in metrics])), world_size, device) for k in metrics[0]}
    if rank == 0:
        logger.info(f'Validation epoch {epoch}: dice={avg_metrics["dice"]:.4f}, iou={avg_metrics["iou"]:.4f}, acc={avg_metrics["accuracy"]:.4f}')
    return avg_metrics


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice, metrics, args):
    torch.save({
        'epoch': epoch,
        'model_state_dict': unwrap(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_iteration': scheduler.current_iteration,
        'best_dice': best_dice,
        'metrics': metrics,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }, path)


def main():
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    set_seed(args.seed, rank)
    device = torch.device(f'cuda:{local_rank if world_size > 1 else args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if world_size > 1:
        holder = [date_str]
        dist.broadcast_object_list(holder, src=0)
        date_str = holder[0]
    run_name = (f'{args.filling_weight}filling_{args.reg_weight}reg_{args.batch_size}bs_{args.epochs}epoch_'
                f'{args.lr}lr_{args.frame_num}frames_{date_str}{args.tag}')
    result_dir = os.path.join(args.output_dir, os.path.basename(os.path.normpath(args.data_dir)), run_name)
    checkpoint_dir = os.path.join(result_dir, 'checkpoints')
    if rank == 0:
        for d in ('checkpoints', 'logs', 'runs'):
            os.makedirs(os.path.join(result_dir, d), exist_ok=True)
        shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(result_dir, 'code'),
                        ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    if world_size > 1:
        dist.barrier()
    logger = setup_logging(os.path.join(result_dir, 'logs', f'training_{date_str}.log'), rank)
    writer = SummaryWriter(os.path.join(result_dir, 'runs')) if rank == 0 else None
    if rank == 0:
        logger.info(f'Result directory: {result_dir}')
        logger.info(f'Arguments: {vars(args)}')
        logger.info(f'Device: {device}, world size: {world_size}')

    train_set = DSASequenceDataset(args.data_dir, 'train', args.frame_num, augmentation=args.augmentation)
    val_set = DSASequenceDataset(args.data_dir, args.val_split, args.frame_num, augmentation=False)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    model = MotionUNet(deep_supervision=args.deep_supervision).to(device)
    if args.init_weights and not args.resume:
        state = torch.load(args.init_weights, map_location=device)
        state = clean_state_dict(state.get('model_state_dict', state))
        result = model.load_state_dict(state, strict=False)
        if rank == 0:
            logger.info(f'Initialised from {args.init_weights}: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys')
    if world_size > 1:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == 'cuda' else {}
        model = DDP(model, **ddp_kwargs)

    seg_loss = SegmentationLoss(args.ce_weight, args.dice_weight, args.focal_weight, args.focal_alpha, args.focal_gamma)
    motion_loss = MotionLoss(similarity_weight=1.0, smoothness_weight=0.05, vessel_threshold=0.5, min_mask_ratio=0.0)
    physics_loss = PhysicsLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = IterationScheduler(optimizer, args.lr, max_iterations=args.epochs * len(train_loader),
                                   warmup_iterations=args.warmup_epochs * len(train_loader), kind=args.scheduler_type,
                                   power=args.poly_power, min_lr_ratio=args.min_lr_ratio)
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == 'cuda' else None

    start_epoch, best_dice = 0, 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        unwrap(model).load_state_dict(clean_state_dict(checkpoint['model_state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', checkpoint['metrics']['dice'])
        scheduler.current_iteration = checkpoint.get('scheduler_iteration', start_epoch * len(train_loader))
        if rank == 0:
            logger.info(f'Resumed from {args.resume} at epoch {start_epoch} (best dice so far {best_dice:.4f})')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_metrics = train_epoch(model, train_loader, seg_loss, motion_loss, physics_loss, optimizer, scheduler,
                                                scaler, device, epoch, args, logger, rank, world_size)
        val_metrics = validate_epoch(model, val_loader, device, epoch, logger, rank, world_size)
        is_best = val_metrics['dice'] > best_dice
        best_dice = max(best_dice, val_metrics['dice'])
        if rank == 0:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('dice/train', train_metrics['dice'], epoch)
            writer.add_scalar('dice/val', val_metrics['dice'], epoch)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            save_checkpoint(os.path.join(checkpoint_dir, 'latest_model.pth'), model, optimizer, scheduler, epoch, best_dice, val_metrics, args)
            if is_best:
                save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pth'), model, optimizer, scheduler, epoch, best_dice, val_metrics, args)
                logger.info(f'New best model: dice={best_dice:.4f}')
            logger.info(f'Epoch {epoch} finished in {time.time() - t0:.1f}s')
    if rank == 0:
        logger.info(f'Training finished, best validation dice: {best_dice:.4f}')
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
