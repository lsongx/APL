import sys
import os
import lmdb 
from PIL import Image
import numpy as np


class Datum(object):
    def __init__(self, shape=None, image=None, label=None):
        self.shape = shape
        self.image = image
        self.label = label

    def SerializeToString(self):
        shape_data = np.asarray(self.shape, dtype=np.uint16).tobytes()
        self.image = self.image.astype(np.uint8)
        self.label = self.label.astype(np.uint8)
        image_data = np.concatenate(
            (self.image.flat, self.label.flat),
            axis=0
        ).tobytes()
        return shape_data+image_data

    def ParseFromString(self, raw_data):
        shape_data = raw_data[:2*3]
        self.shape = np.frombuffer(shape_data, dtype=np.uint16).astype(np.uint64)
        
        raw_img_data = raw_data[2*3:]
        total_data = np.frombuffer(raw_img_data, dtype=np.uint8)
        image_size = self.shape[0]*self.shape[1]*self.shape[2]
        self.image = total_data[:image_size]
        self.image = self.image.reshape(self.shape)
        self.label = total_data[image_size:]
        self.label = self.label.reshape(self.shape[:2])



def get_path_by_dataset(folder, name, set_name):
    path_map_by_set = {
        'gta5': ('images', 'labels'),
        'cityscapes_train': ('leftImg8bit/train', 'gtFine/train'),
        'cityscapes_val': ('leftImg8bit/val', 'gtFine/val'),
        'synthia': ('images', 'labels'),
    }
    image_sub, label_sub = path_map_by_set[set_name]
    image_path = os.path.join(folder, image_sub, name)
    if 'cityscapes' in set_name:
        name = name.replace('leftImg8bit', 'gtFine_labelIds')
        label_path = os.path.join(folder, label_sub, name)
    return image_path, label_path


def create_dataset(output_path, image_folder, image_list, set_name):
    image_name_list = [i.strip() for i in open(image_list)]
    n_samples = len(image_name_list)
    env = lmdb.open(output_path, map_size=1099511627776) # 1TB
    cache = {}

    with env.begin(write=True) as txn:
        # for idx, image_name in tqdm(enumerate(image_name_list)):
        for idx, image_name in enumerate(image_name_list):
            image_path, label_path = get_path_by_dataset(image_folder, image_name, set_name)

            if not os.path.isfile(image_path):
                raise RuntimeError('%s does not exist' % image_path)
            image = np.asarray(Image.open(image_path).convert('RGB'))
            label = np.asarray(Image.open(label_path))
            assert image.shape[0] == label.shape[0]
            assert image.shape[1] == label.shape[1]

            shape = image.shape
            datum = Datum(image.shape, image, label)

            txn.put(image_name.encode('ascii'), datum.SerializeToString())

    print(f'Created dataset with {n_samples:d} samples')


if __name__ == '__main__':
    # output_path = '/mnt/data-1/data/liangchen.song/seg/lmdb_data/gta5_valid'
    # image_folder = '/mnt/data-1/data/liangchen.song/seg/ori_gta_trans'
    # image_list = '/mnt/data-1/data/liangchen.song/seg/ori_gta_trans/valid_imagelist.txt'
    # create_dataset(output_path, image_folder, image_list, 'gta5')

    # output_path = '/mnt/data-1/data/liangchen.song/seg/lmdb_data/cityscapes_train'
    # image_folder = '/mnt/data-1/data/liangchen.song/seg/cityscapes'
    # image_list = '/mnt/data-1/data/liangchen.song/seg/cityscapes_train.txt'
    # create_dataset(output_path, image_folder, image_list, 'cityscapes_train')

    output_path = '/mnt/data-1/data/liangchen.song/seg/lmdb_data/cityscapes_val'
    image_folder = '/mnt/data-1/data/liangchen.song/seg/cityscapes'
    image_list = '/mnt/data-1/data/liangchen.song/seg/cityscapes_val.txt'
    create_dataset(output_path, image_folder, image_list, 'cityscapes_val')
