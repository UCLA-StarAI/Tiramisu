import os
import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_img_fnames(data_root, name, data_paths = list(), visited = set()):
    if name in visited:
        return

    visited.add(name)

    full_name = os.path.join(data_root, name)
    for ch_name in os.listdir(full_name):
        if os.path.isfile(os.path.join(full_name, ch_name)):
            data_paths.append(os.path.join(name, ch_name))
        elif os.path.isdir(os.path.join(full_name, ch_name)):
            get_img_fnames(data_root, os.path.join(name, ch_name), data_paths, visited)

    print(len(data_paths))

    return


class LSUNBase(Dataset):
    def __init__(self, data_root, train = True, size=None,
                 interpolation="bicubic",
                 flip_p=0.0
                 ):
        if train:
            self.data_root = os.path.join(data_root, "train/")
            self.index_fname = os.path.join(data_root, "train_fnames.txt")
        else:
            self.data_root = os.path.join(data_root, "val/")
            self.index_fname = os.path.join(data_root, "val_fnames.txt")
        
        if os.path.exists(self.index_fname):
            with open(self.index_fname, "r") as f:
                self.data_paths = f.read().splitlines()
        else:
            self.data_paths = list()
            get_img_fnames(self.data_root, name = "./", data_paths = self.data_paths)

            with open(self.index_fname, "w") as f:
                for data_path in self.data_paths:
                    f.write(f"{data_path}\n")

        self.image_paths = self.data_paths

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.BILINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/scratch/anji/data/LSUN/bedrooms", train = True, **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="/scratch/anji/data/LSUN/bedrooms", train = False, **kwargs)


class LSUNBedroomsWithMask(LSUNBase):
    def __init__(self, root = "/scratch/anji/data/LSUN/", mask_type = "left", train = True, **kwargs):
        super().__init__(data_root=os.path.join(root, "bedrooms/"), train = train, **kwargs)

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
        example = super(LSUNBedroomsWithMask, self).__getitem__(i)
        image = example["image"]
        img_name = f"image_{i}"

        if self.mask is not None:
            return torch.from_numpy(image).permute(2, 0, 1), self.mask, img_name
        else:
            mask_fname = self.mask_fnames[i % len(self.mask_fnames)]
            mask = torch.from_numpy(np.array(Image.open(mask_fname)).sum(axis = 2) / 765.0).float().unsqueeze(0)
            
            return image, mask, img_name


class LSUNBedroomsWithMultiMask(LSUNBase):
    def __init__(self, root = "/scratch/anji/data/LSUN/", mask_type = "left", train = True, **kwargs):
        super().__init__(data_root=os.path.join(root, "bedrooms/"), train = train, **kwargs)

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