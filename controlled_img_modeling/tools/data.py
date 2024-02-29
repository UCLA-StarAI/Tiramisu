import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
import collections
import sys

sys.path.append("../../../external/taming-transformers")
sys.path.append("../../external/taming-transformers")

from taming.data.helper_types import Annotation
import os

from tools.utils import instantiate_from_config


def custom_collate(batch):
    r"""source: pytorch 1.9.0, only one modification to original code """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, collections.abc.Sequence) and isinstance(elem[0], Annotation):  # added
        return batch  # added
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, drop_last=False, shuffle_train=True, 
                 shuffle_validation=False, transform_fns=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self.shuffle_validation = shuffle_validation

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self, sampler = None):
        if sampler is None:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=self.shuffle_train, collate_fn=custom_collate,
                            drop_last=self.drop_last)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                            num_workers=self.num_workers, sampler = sampler, collate_fn=custom_collate,
                            drop_last=self.drop_last)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate,
                          drop_last=self.drop_last, shuffle=self.shuffle_validation)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate,
                          drop_last=self.drop_last, shuffle=False)


def get_input(batch, key):
    if key == "none":
        for key in batch.keys():
            B = batch[key].size(0)
            break
        return torch.zeros([B])
    x = batch[key]
    if len(x.shape) == 3:
        x = x[..., None]
    if len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    if x.dtype == torch.double:
        x = x.float()
    return x


class ZCNumpyDataset(Dataset):
    def __init__(self, data_path, split, key_mapping = None, num_samples_per_epoch = None):
        super(ZCNumpyDataset, self).__init__()

        self.data_path = data_path
        self.split = split
        self.num_samples_per_epoch = num_samples_per_epoch

        with open(os.path.join(data_path, "summary.txt"), "r") as f:
            lines = f.readlines()
            self.num_c_vars = int(lines[2].split(" ")[-1].strip("\n"))
            self.num_z_vars = int(lines[3].split(" ")[-1].strip("\n"))

        data = np.load(os.path.join(data_path, "latents.npz"))
        self.data_dict = dict()
        if split == "train":
            for key in data.keys():
                if key.startswith("tr_"):
                    k = key[3:]
                    if key_mapping is not None and k in key_mapping:
                        k = key_mapping[k]
                    self.data_dict[k] = torch.from_numpy(data[key])
        elif split == "validation" or split == "test":
            for key in data.keys():
                if key.startswith("vl_"):
                    k = key[3:]
                    if key_mapping is not None and k in key_mapping:
                        k = key_mapping[k]
                    self.data_dict[k] = torch.from_numpy(data[key])
        else:
            raise ValueError()

        for k, v in self.data_dict.items():
            self.length = v.size(0)

        self._count = 0
        self._shuffle = None

    def __len__(self):
        if self.num_samples_per_epoch is None:
            return self.length
        else:
            return self.num_samples_per_epoch

    def __getitem__(self, idx):
        if self.num_samples_per_epoch is None:
            return {key: value[idx] for key, value in self.data_dict.items()}
        else:
            if self._shuffle is None:
                self._shuffle = torch.randperm(self.length)[:self.num_samples_per_epoch]
            elif self._count >= self.num_samples_per_epoch:
                self._shuffle = torch.randperm(self.length)[:self.num_samples_per_epoch]
                # pass

            self._count += 1

            idx = self._shuffle[idx]

            return {key: value[idx] for key, value in self.data_dict.items()}