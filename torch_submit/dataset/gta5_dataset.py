import os
import os.path as osp
import numpy as np
import time
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import lmdb
from .lmdb_dataset import Datum


class GTA5DataSet(data.Dataset):
    def __init__(
        self, root, list_path, joint_transform=None, sliding_crop=None, 
        transform=None, target_transform=None, ignore_label=255
    ):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.sliding_crop = sliding_crop
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["img"]

        # re-assign labels to match the format of Cityscapes
        label = np.asarray(label, np.float32)
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        mask = Image.fromarray(label_copy.astype(np.uint8))
        img = image

        if self.joint_transform is not None:
            try:
                img, mask = self.joint_transform(img, mask)
            except:
                print(name)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info), name
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return img, mask, name


class GTA5DataSetLMDB(data.Dataset):
    def __init__(
        self, root, list_path, joint_transform=None, sliding_crop=None,
        transform=None, target_transform=None, ignore_label=255
    ):
        self.env = lmdb.open(root,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.sliding_crop = sliding_crop
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join("images/%s" % name)
            label_file = osp.join("labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        name = self.files[index]['name']

        datum = Datum()
        data_bin = self.txn.get(name.encode('ascii'))
        if data_bin is None:
            raise RuntimeError(f'Key {name} not found.')
        datum.ParseFromString(data_bin)

        label = np.asarray(datum.label, np.float32)
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        mask = Image.fromarray(label_copy.astype(np.uint8))
        img = Image.fromarray(datum.image)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info), name
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return img, mask, name

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
