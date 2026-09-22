import argparse
import csv
import os

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dsa_dataset import INTENSITY_MEAN, INTENSITY_STD, DSASequenceDataset
from models import MotionUNet
from utils.common import clean_state_dict
from utils.metrics import case_metrics

METRICS = ('dice', 'iou', 'precision', 'recall', 'asd')


def parse_args():
    p = argparse.ArgumentParser(description='Inference and evaluation on one split')
    p.add_argument('--model_path', type=str, required=True, help='run directory (uses checkpoints/best_model.pth) or a checkpoint file')
    p.add_argument('--data_dir', type=str, default='./data/DSA_sequences')
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--frame_num', type=int, default=None, help='defaults to the value stored in the checkpoint')
    p.add_argument('--output_dir', type=str, default=None, help='defaults to <run directory>/inference')
    p.add_argument('--save_inputs', action='store_true', help='also save the (de-normalised) input sequences')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.isdir(args.model_path):
        run_dir, ckpt_path = args.model_path, os.path.join(args.model_path, 'checkpoints', 'best_model.pth')
    else:
        run_dir, ckpt_path = os.path.dirname(os.path.dirname(args.model_path)), args.model_path
    output_dir = args.output_dir or os.path.join(run_dir, 'inference')
    pred_dir = os.path.join(output_dir, args.split)
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_args = checkpoint.get('args', {})
    frame_num = args.frame_num or saved_args.get('frame_num', 6)
    model = MotionUNet(deep_supervision=True)
    model.load_state_dict(clean_state_dict(checkpoint['model_state_dict']))
    model.to(device).eval()
    print(f'Loaded {ckpt_path} ({frame_num} frames, epoch {checkpoint.get("epoch")})')

    dataset = DSASequenceDataset(args.data_dir, args.split, frame_num, augmentation=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Inference on {args.split}'):
            images = batch['image'].to(device)
            name = batch['file_name'][0]
            logits = model(images)['predictions']
            if isinstance(logits, list):
                logits = logits[0]
            pred = torch.argmax(logits, dim=2)[0].cpu().numpy().astype(np.uint8)
            gt = batch['mask'][0].numpy().astype(np.uint8)
            m = case_metrics(pred[-1] > 0, gt[-1] > 0)
            m['file_name'] = name
            rows.append(m)
            sitk.WriteImage(sitk.GetImageFromArray(pred), os.path.join(pred_dir, f'{name}_pred.nii.gz'))
            sitk.WriteImage(sitk.GetImageFromArray(gt), os.path.join(pred_dir, f'{name}_gt.nii.gz'))
            if args.save_inputs:
                inputs = (images[0, :, 0].cpu().numpy() * INTENSITY_STD + INTENSITY_MEAN).astype(np.float32)
                sitk.WriteImage(sitk.GetImageFromArray(inputs), os.path.join(pred_dir, f'{name}_input.nii.gz'))

    with open(os.path.join(output_dir, f'metrics_{args.split}.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'{len(rows)} cases from split "{args.split}", checkpoint {ckpt_path}']
    for k in METRICS:
        values = np.array([r[k] for r in rows], dtype=float)
        lines.append(f'{k:>9}: {values.mean():.4f} ({values.std():.4f})')
    summary = '\n'.join(lines)
    with open(os.path.join(output_dir, f'metrics_{args.split}.txt'), 'w') as f:
        f.write(summary + '\n')
    print(summary)
    print(f'Predictions saved to {pred_dir}')


if __name__ == '__main__':
    main()
