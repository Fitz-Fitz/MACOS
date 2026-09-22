import json
import os

import albumentations as A
import cv2
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

# Intensity statistics of the training set used for z-score normalisation (raw 8-bit DSA intensities).
INTENSITY_MEAN = 106.76818084716797
INTENSITY_STD = 21.11110496520996


class DSASequenceDataset(Dataset):
    """One DSA injection sequence per item.

    Expected layout: <base_dir>/data_splits.json with {"train": [...], "test": [...]} listing sequence
    folder names, and <base_dir>/<name>/{video.nii.gz, mask.nii.gz}, both of shape (T, H, W).
    The last frame is the key frame; only its mask is used for supervision.

    Returns {'image': (T', 1, H', W') float tensor, 'mask': (T', H', W') float tensor, 'file_name': str}
    where T' = frame_num frames are sampled uniformly over the sequence (first and last frame always kept).
    """

    def __init__(self, base_dir, split='train', frame_num=6, augmentation=False, image_size=(256, 256)):
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f'Dataset directory {base_dir} does not exist')
        self.base_dir = base_dir
        self.split = split
        self.frame_num = frame_num
        self.image_size = tuple(image_size)
        self.names = self._read_split(base_dir, split)
        self.augmentation = augmentation and split == 'train'
        print(f'{split} split: {len(self.names)} sequences, {frame_num} frames, resized to {self.image_size}')

        if self.augmentation:
            self.geometric_transform = A.Compose([
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ElasticTransform(p=0.5),
                A.GridDistortion(p=0.5),
            ])
            # photometric transforms operate on frames normalised to [0, 1]
            self.photometric_transform = A.Compose([
                A.GaussNoise(var_limit=(0.0, 0.005), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ])

    @staticmethod
    def _read_split(base_dir, split):
        splits_file = os.path.join(base_dir, 'data_splits.json')
        if not os.path.isfile(splits_file):
            raise FileNotFoundError(f'Split file {splits_file} does not exist')
        with open(splits_file) as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"Split '{split}' not found in {splits_file}, available: {sorted(splits)}")
        return splits[split]

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _load(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} does not exist')
        return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

    def _sample_frames(self, video, mask):
        indices = np.linspace(0, video.shape[0] - 1, self.frame_num, dtype=int)
        return video[indices].copy(), mask[indices].copy()

    def _augment(self, video, mask):
        # albumentations treats the temporal axis as channels, so every frame gets the same geometric transform
        out = self.geometric_transform(image=video.transpose(1, 2, 0), mask=mask.transpose(1, 2, 0))
        video, mask = out['image'].transpose(2, 0, 1), out['mask'].transpose(2, 0, 1)
        lo, hi = video.min(), video.max()
        video = (video - lo) / (hi - lo + 1e-8)
        for t in range(video.shape[0]):
            video[t] = self.photometric_transform(image=video[t])['image']
        video = video * (hi - lo) + lo
        return video, mask

    def _resize(self, video, mask):
        H, W = self.image_size
        if video.shape[1:] == (H, W):
            return video, mask
        video_r = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in video])
        mask_r = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_NEAREST) for f in mask])
        return video_r, mask_r

    def __getitem__(self, idx):
        name = self.names[idx]
        video = self._load(os.path.join(self.base_dir, name, 'video.nii.gz'))
        mask = self._load(os.path.join(self.base_dir, name, 'mask.nii.gz'))
        if video.ndim == 2:
            video, mask = video[None], mask[None]
        video, mask = self._sample_frames(video, mask)
        if self.augmentation:
            video, mask = self._augment(video, mask)
        video = (video - INTENSITY_MEAN) / (INTENSITY_STD + 1e-8)
        mask = (mask > 0).astype(np.uint8)
        video, mask = self._resize(video, mask)
        return {
            'image': torch.from_numpy(video.astype(np.float32)).unsqueeze(1),
            'mask': torch.from_numpy(mask.astype(np.float32)),
            'file_name': name,
        }
