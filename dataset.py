import os
import sys
import random

import numpy as np
from PIL import Image
import h5py
from glob import glob
from tqdm import tqdm


class Dataset():

    def __init__(self, h5_path, mode, img_width, img_height, img_dim, is_mem):
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height
        self.img_dim = img_dim
        self.is_mem = is_mem

        assert mode == 'w' or mode == 'r', 'mode must be \'w\' or \'r\''
        if self.mode == 'w':
            if os.path.exists(h5_path):
                while True:
                    inp = input('overwrite {}? (y/n)\n'.format(h5_path))
                    if inp == 'y' or inp == 'n':
                        break
                if inp == 'n':
                    print('canceled')
                    sys.exit()
            self.h5file = h5py.File(h5_path, mode)
        if self.mode == 'r':
            assert os.path.exists(h5_path), 'hdf5 file is not found: {}'.format(h5_path)
            self.h5file = h5py.File(h5_path, mode)
            if self.is_mem:
                self._get = self._get_from_mem
            else:
                self._get = self._get_from_file

    def load_imgs(self, src_dir_path):
        dir_paths = sorted(glob('{}/*'.format(src_dir_path)))
        for dir_path in tqdm(dir_paths):
            if not os.path.isdir(dir_path):
                continue
            img_paths = sorted(glob('{}/**/*.png'.format(dir_path)))
            imgs = np.empty((len(img_paths), self.img_width, self.img_height, self.img_dim), dtype=np.float32)
            for i, img_path in enumerate(img_paths):
                pil_img = Image.open(img_path)
                np_img = np.asarray(pil_img)
                np_img = (np_img.astype(np.float32) / 127.5) - 1.
                if len(np_img.shape) == 2:
                    np_img = np_img[np.newaxis, :, :, np.newaxis]
                    if self.img_dim == 3:
                        np_img = np.repeat(np_img, 3, axis=3)
                elif len(np_img.shape) == 3:
                    np_img = np_img[np.newaxis, :, :, :]
                imgs[i] = np_img
            self._save(os.path.basename(dir_path), imgs)

    def _save(self, group_name, imgs):
        self.h5file.create_group(group_name)
        self.h5file.create_dataset(group_name + '/imgs', data=imgs)
        self.h5file.flush()

    def set_load_data(self, train_rate=1.):
        print('preparing dataset...')
        self.keys_queue = list()
        self.label_to_id = dict()
        for i, (key, val) in enumerate(self.h5file.items()):
            img_n = len(val['imgs'])
            for j in range(img_n):
                self.keys_queue.append((key, j))
            self.label_to_id[key] = i
        self.label_n = len(self.label_to_id)
        if self.is_mem:
            self._put_on_mem()

    def shuffle(self):
        random.shuffle(self.keys_queue)

    def get_data_n(self):
        return len(self.keys_queue)

    def get_data_n_by_labels(self, labels):
        filtered_keys_queue = list(filter(lambda x: x[0] in labels, self.keys_queue))
        return len(filtered_keys_queue)

    def get_ids_from_labels(self, labels):
        ids = list()
        for label in labels:
            ids.append(self.label_to_id[label])
        return ids

    def get_labels(self):
        labels = list()
        for key in self.label_to_id.keys():
            labels.append(key)
        return labels

    def get_batch(self, batch_i, batch_size):
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            keys_list.append(self.keys_queue[i])
        return self._get(keys_list)

    def get_batch_by_labels(self, batch_i, batch_size, labels):
        filtered_keys_queue = list(filter(lambda x: x[0] in labels, self.keys_queue))
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            keys_list.append(filtered_keys_queue[i])
        return self._get(keys_list)

    def get_random(self, batch_size):
        keys_list = list()
        for _ in range(batch_size):
            keys_list.append(random.choice(self.keys_queue))
        return self._get(keys_list)

    def get_random_by_labels(self, batch_size, labels):
        filtered_keys_queue = list(filter(lambda x: x[0] in labels, self.keys_queue))
        keys_list = list()
        for _ in range(batch_size):
            keys_list.append(random.choice(filtered_keys_queue))
        return self._get(keys_list)

    def _get_from_file(self, keys_list):
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        for i, keys in enumerate(keys_list):
            img = self.h5file[keys[0] + '/imgs'].value[keys[1]]
            imgs[i] = img[np.newaxis, :]
        return imgs

    def _put_on_mem(self):
        print('putting data on memory...')
        self.imgs = np.empty((self.label_n, self.img_n, self.img_width, self.img_height, self.img_dim), np.float32)
        self.label_to_img_n = dict()
        for i, key in enumerate(self.h5file.keys()):
            val = self.h5file[key + '/imgs'].value
            if len(val) < self.img_n:
                white_imgs = np.ones((self.img_n - len(val), self.img_width, self.img_height, self.img_dim), np.float32)
                val = np.concatenate((val, white_imgs), axis=0)
            self.imgs[i] = val
            self.label_to_img_n[key] = len(self.imgs[i])

    def _get_from_mem(self, keys_list):
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        for i, keys in enumerate(keys_list):
            assert keys[1] < self.label_to_img_n[keys[0]], 'Image is out of range'
            img = self.imgs[self.label_to_id[keys[0]]][keys[1]]
            imgs[i] = img[np.newaxis, :]
        return imgs
