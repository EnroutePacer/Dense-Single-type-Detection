"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
import math
from functools import partial

from ..core import register
torchvision.disable_beta_transforms_warning()
from copy import deepcopy
from PIL import Image, ImageDraw
import os


__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        stop_epoch=None, 
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        scales=None,
        scale_weights=None,
        final_size=None,
        mixup_prob=0.0,
        mixup_epochs=[0, 0],
        data_vis=False,
        vis_save='./vis_dataset/'
    ) -> None:
        super().__init__()
        self.base_size = base_size
        if scales is not None and base_size_repeat is not None:
            raise ValueError("Use either explicit scales or base_size_repeat, not both")
        if scales is not None:
            self.scales = [int(size) for size in scales]
            if not self.scales:
                raise ValueError("scales must contain at least one input size")
            if any(size <= 0 or size % 32 != 0 for size in self.scales):
                raise ValueError("Every multi-scale input size must be positive and divisible by 32")
        else:
            self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        if scale_weights is not None:
            if self.scales is None:
                raise ValueError("scale_weights requires explicit scales")
            self.scale_weights = [float(weight) for weight in scale_weights]
            if len(self.scale_weights) != len(self.scales):
                raise ValueError("scale_weights must have the same length as scales")
            if any(not math.isfinite(weight) or weight < 0 for weight in self.scale_weights):
                raise ValueError("scale_weights must contain finite, non-negative values")
            total_weight = sum(self.scale_weights)
            if total_weight <= 0:
                raise ValueError("scale_weights must contain at least one positive value")
            self.scale_weights = [weight / total_weight for weight in self.scale_weights]
        else:
            self.scale_weights = None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.final_size = int(final_size) if final_size is not None else None
        if self.final_size is not None and (self.final_size <= 0 or self.final_size % 32 != 0):
            raise ValueError("final_size must be positive and divisible by 32")
        self.ema_restart_decay = ema_restart_decay
        # FIXME Mixup
        self.mixup_prob, self.mixup_epochs = mixup_prob, mixup_epochs
        if self.mixup_prob > 0:
            self.data_vis, self.vis_save = data_vis, vis_save
            os.makedirs(self.vis_save, exist_ok=True) if self.data_vis else None
            print("     ### Using MixUp with Prob@{} in {} epochs ### ".format(self.mixup_prob, self.mixup_epochs))
        if self.scales is not None:
            print("     ### Multi-scale Training until {} epochs ### ".format(self.stop_epoch))
            print("     ### Multi-scales@ {} ###        ".format(self.scales))
            if self.scale_weights is not None:
                print("     ### Multi-scale weights@ {} ###        ".format(
                    list(zip(self.scales, self.scale_weights))))
            if self.final_size is not None:
                print("     ### Fixed training size after multi-scale@ {} ###        ".format(self.final_size))
        self.print_info_flag = True
        # self.interpolation = interpolation

    def apply_mixup(self, images, targets):
        """
        Applies Mixup augmentation to the batch if conditions are met.

        Args:
            images (torch.Tensor): Batch of images.
            targets (list[dict]): List of target dictionaries corresponding to images.

        Returns:
            tuple: Updated images and targets
        """
        # Log when Mixup is permanently disabled
        if self.epoch == self.mixup_epochs[-1] and self.print_info_flag:
            print(f"     ### Attention --- Mixup is closed after epoch@ {self.epoch} ###")
            self.print_info_flag = False

        # Apply Mixup if within specified epoch range and probability threshold
        if random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]:
            # Generate mixup ratio
            beta = round(random.uniform(0.45, 0.55), 6)

            # Mix images
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

            # Prepare targets for Mixup
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)

            for i in range(len(targets)):
                # Combine boxes, labels, and areas from original and shifted targets
                updated_targets[i]['boxes'] = torch.cat([targets[i]['boxes'], shifted_targets[i]['boxes']], dim=0)
                updated_targets[i]['labels'] = torch.cat([targets[i]['labels'], shifted_targets[i]['labels']], dim=0)
                updated_targets[i]['area'] = torch.cat([targets[i]['area'], shifted_targets[i]['area']], dim=0)

                # Add mixup ratio to targets
                updated_targets[i]['mixup'] = torch.tensor(
                    [beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']), 
                    dtype=torch.float32
                    )
            targets = updated_targets

            if self.data_vis:
                for i in range(len(updated_targets)):
                    image_tensor = images[i]
                    image_tensor_uint8 = (image_tensor * 255).type(torch.uint8)
                    image_numpy = image_tensor_uint8.numpy().transpose((1, 2, 0))
                    pilImage = Image.fromarray(image_numpy)
                    draw = ImageDraw.Draw(pilImage)
                    print('mix_vis:', i, 'boxes.len=', len(updated_targets[i]['boxes']))
                    for box in updated_targets[i]['boxes']:
                        draw.rectangle([int(box[0]*640 - (box[2]*640)/2), int(box[1]*640 - (box[3]*640)/2), 
                                        int(box[0]*640 + (box[2]*640)/2), int(box[1]*640 + (box[3]*640)/2)], outline=(255,255,0))
                    pilImage.save(self.vis_save + str(i) + "_"+ str(len(updated_targets[i]['boxes'])) +'_out.jpg')

        return images, targets

    def __call__(self, items):
        images = [x[0] for x in items]
        targets = [x[1] for x in items]

        # Choose the batch scale before stacking.  This permits the dataset
        # pipeline to keep every image at its original resolution and makes
        # multi-scale training a single interpolation (original -> sz).
        sz = None
        if self.scales is not None:
            if self.epoch < self.stop_epoch:
                sz = random.choices(self.scales, weights=self.scale_weights, k=1)[0] \
                    if self.scale_weights is not None else random.choice(self.scales)
            else:
                sz = self.final_size

        if sz is not None:
            resized_images = []
            for image in images:
                if tuple(image.shape[-2:]) != (sz, sz):
                    image = F.interpolate(
                        image[None],
                        size=(sz, sz),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True,
                    )[0]
                resized_images.append(image)
            images = torch.stack(resized_images, dim=0)

            if sz is not None and 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=(sz, sz), mode='nearest')
                raise NotImplementedError('')
        else:
            image_shapes = {tuple(image.shape) for image in images}
            if len(image_shapes) != 1:
                raise ValueError(
                    'Cannot batch variable-size images without a configured multi-scale size'
                )
            images = torch.stack(images, dim=0)

        # MixUp requires a dense batch, so it must run after the one-pass resize.
        images, targets = self.apply_mixup(images, targets)

        return images, targets
