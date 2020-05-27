import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import lmdb
from .lmdb_dataset import Datum


class CityscapesDataSet(data.Dataset):
    def __init__(
        self, root, list_path, 
        joint_transform=None, sliding_crop=None, transform=None, target_transform=None,
    ):
        self.root = root
        self.list_path = list_path
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        if 'val' in list_path:
            self.train_set = 'val'
        elif 'test' in list_path:
            self.train_set = 'test'
        else:
            self.train_set = 'train'
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.train_set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.train_set, name[:-15]+'gtFine_labelIds.png'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name.split('/')[-1]
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        if self.train_set != 'test':
            label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        # image = image.resize(self.crop_size, Image.BICUBIC)
        # if self.train_set == 'train':
        #     label = label.resize(self.crop_size, Image.NEAREST)

        # image = np.asarray(image, np.float32)
        # label = np.asarray(label, np.float32)

        # label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        # for k, v in self.id_to_trainid.items():
        #     label_copy[label == k] = v

        # size = image.shape
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        # image = image.transpose((2, 0, 1))

        # return image.copy(), label_copy.copy(),np.array(size), name
        if self.train_set != 'test':
            label = np.asarray(label, np.float32)
            label_copy = 255 * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            mask = Image.fromarray(label_copy.astype(np.uint8))
        else:
            label_copy = 255 * np.ones([512, 1024], dtype=np.float32)
            mask = Image.fromarray(label_copy.astype(np.uint8))
        img = image

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


class CityscapesDataSetLMDB(data.Dataset):
    def __init__(
        self, root, list_path,
        joint_transform=None, sliding_crop=None, transform=None, target_transform=None,
    ):
        self.root = root
        self.env = lmdb.open(root,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.txn = self.env.begin(write=False)
        self.list_path = list_path
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        if 'val' in list_path:
            self.train_set = 'val'
        else:
            self.train_set = 'train'
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" %
                                (self.train_set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" %
                                  (self.train_set, name[:-15]+'gtFine_labelIds.png'))
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
