import random
from PIL import Image
import torchvision.transforms.functional as TF
import torch


class PairedTransforms:
    def __init__(self, size=(256,256), is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img_ir, img_vis, mask=None):
        # img_ir, img_vis: PIL Images; mask: PIL Image or None
        # random flip
        if self.is_train and random.random() > 0.5:
            img_ir = TF.hflip(img_ir)
            img_vis = TF.hflip(img_vis)
            if mask: mask = TF.hflip(mask)
        # random rotation
        if self.is_train and random.random() > 0.7:
            angle = random.uniform(-15,15)
            img_ir = TF.rotate(img_ir, angle)
            img_vis = TF.rotate(img_vis, angle)
            if mask: mask = TF.rotate(mask, angle)
        # resize
        img_ir = TF.resize(img_ir, self.size)
        img_vis = TF.resize(img_vis, self.size)
        if mask: mask = TF.resize(mask, self.size)
        # to tensor
        t_ir = TF.to_tensor(img_ir)
        t_vis = TF.to_tensor(img_vis)
        # normalize to mean 0.5 std 0.5
        t_ir = TF.normalize(t_ir, [0.5,0.5,0.5], [0.5,0.5,0.5])
        t_vis = TF.normalize(t_vis, [0.5,0.5,0.5], [0.5,0.5,0.5])
        if mask:
            t_mask = TF.to_tensor(mask)
            # ensure single channel
            if t_mask.shape[0] > 1:
                t_mask = t_mask.mean(dim=0, keepdim=True)
            return t_ir, t_vis, t_mask
        return t_ir, t_vis


def build_transforms(input_size=(256,256), is_train=True):
    return PairedTransforms(size=input_size, is_train=is_train)

