import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image

from .base import ImagePaths


class CelebA256(Dataset):
    def __init__(self, root=None, train=True, num_samples=None):
        super(CelebA256, self).__init__()
        self.root = root
        self.train = train
        self.ns = num_samples

        self._load()

        self.loaded_samples = 0
        self.sshuffle = np.random.permutation(len(self.data))

    def __len__(self):
        l = len(self.data) if self.ns is None else self.ns
        return l

    def __getitem__(self, i):
        if self.ns is not None:
            self.loaded_samples += 1
            if self.loaded_samples >= len(self):
                self.sshuffle = np.random.permutation(len(self.data))
                self.loaded_samples = 0
            return self.data[self.sshuffle[i]]
        else:
            return self.data[i]

    def _load(self):
        if self.train:
            base_dir = os.path.join(self.root, "train256/")
        else:
            base_dir = os.path.join(self.root, "val256/")

        self.abspaths = []
        for fname in os.listdir(base_dir):
            abspath = os.path.join(base_dir, fname)
            if os.path.exists(abspath):
                self.abspaths.append(abspath)

        self.abspaths.sort()

        self.data = ImagePaths(self.abspaths)


class CelebA256WithMask(CelebA256):
    def __init__(self, root=None, train=True, num_samples=None, mask_type="left"):
        super(CelebA256WithMask, self).__init__(root = root, train = train, num_samples = num_samples)

        self.mask_type = mask_type
        self.mask = torch.zeros([1, 256, 256], dtype = torch.float32)
        if self.mask_type == "left":
            self.mask[0,:,:128] = 1.0
        elif self.mask_type == "top":
            self.mask[0,:128,:] = 1.0
        elif self.mask_type == "small_eye":
            self.mask[0,64:128,64:128] = 1.0
        elif self.mask_type == "expand":
            self.mask[0,96:160,96:160] = 1.0
        elif self.mask_type == "single_vert_strip":
            self.mask[0,72:164,120:136] = 1.0
        elif self.mask_type == "single_horiz_strip":
            self.mask[0,120:136,72:164] = 1.0
        elif self.mask_type == "vertical_strips":
            self.mask[0,72:164,56:72] = 1.0
            self.mask[0,72:164,120:136] = 1.0
            self.mask[0,72:164,184:200] = 1.0
        elif self.mask_type == "horizontal_strips":
            self.mask[0,56:72,72:164] = 1.0
            self.mask[0,120:136,72:164] = 1.0
            self.mask[0,184:200,72:164] = 1.0
        elif self.mask_type == "thin":
            self.mask = None
            self.mask_fnames = []
            for fname in os.listdir("./gt_keep_masks/thin/"):
                self.mask_fnames.append(os.path.join("./gt_keep_masks/thin/", fname))
            self.mask_fnames.sort()
        elif self.mask_type == "thick":
            self.mask = None
            self.mask_fnames = []
            for fname in os.listdir("./gt_keep_masks/thick/"):
                self.mask_fnames.append(os.path.join("./gt_keep_masks/thick/", fname))
            self.mask_fnames.sort()
        else:
            raise ValueError()

    def __getitem__(self, i):
        if self.ns is not None:
            self.loaded_samples += 1
            if self.loaded_samples >= len(self):
                self.sshuffle = np.random.permutation(len(self.data))
                self.loaded_samples = 0
            img_name = f"celeba_{self.sshuffle[i]}"
            image = self.data[self.sshuffle[i]]["image"]
            image = torch.from_numpy(image).permute(2, 0, 1)

            i = self.shuffle[i]
            
        else:
            img_name = f"celeba_{i}"
            image = self.data[i]["image"]
            image = torch.from_numpy(image).permute(2, 0, 1)

        if self.mask is not None:
            return image, self.mask, img_name
        else:
            mask_fname = self.mask_fnames[i % len(self.mask_fnames)]
            mask = torch.from_numpy(np.array(Image.open(mask_fname)).sum(axis = 2) / 765.0).float().unsqueeze(0)
            
            return image, mask, img_name


class CelebA256WithMultiMask(CelebA256):
    def __init__(self, root=None, train=True, num_samples=None, mask_type="left-right"):
        super(CelebA256WithMultiMask, self).__init__(root = root, train = train, num_samples = num_samples)

        self.mask_type = mask_type
        self.mask = torch.zeros([1, 256, 256], dtype = torch.long) + 9999
        if self.mask_type == "left-right":
            self.mask[0,:,:128] = 0
            self.mask[0,:,128:] = 1
            self.num_imgs = 2
        elif self.mask_type == "two-windows1":
            self.mask[0,64:128,32:96] = 0
            self.mask[0,160:224,96:160] = 1
            self.num_imgs = 2
        elif self.mask_type == "two-windows2":
            self.mask[0,32:96,64:192] = 0
            self.mask[0,160:224,64:192] = 1
            self.num_imgs = 2
        elif self.mask_type == "two-windows3":
            self.mask[0,64:192,32:96] = 0
            self.mask[0,64:192,160:224] = 1
            self.num_imgs = 2
        elif self.mask_type == "up-down":
            self.mask[0,:128,:] = 0
            self.mask[0,128:,:] = 1
            self.num_imgs = 2
        elif self.mask_type == "eye-mouth":
            self.mask[0,64:128,64:128] = 0
            self.mask[0,176:208,96:160] = 1
            self.num_imgs = 2
        elif self.mask_type == "upper-lower":
            self.mask[0,0:160,0:64] = 0
            self.mask[0,0:64,:] = 0
            self.mask[0,0:160,192:256] = 0
            self.mask[0,176:208,96:160] = 1
            self.num_imgs = 2
        else:
            raise ValueError()

    def __len__(self):
        return super().__len__() // self.num_imgs

    def __getitem__(self, i):

        img_name = f"celeba_{i}"

        imgs = []
        for j in range(self.num_imgs):
            img = super().__getitem__(i * self.num_imgs + j)["image"]
            img = torch.from_numpy(img).permute(2, 0, 1)
            imgs.append(img)

        imgs = torch.stack(imgs, dim = 0)

        return imgs, self.mask, img_name