import json
import cv2
import os
import numpy as np
from PIL import Image
import random
import torch
import re

from torch.utils.data import Dataset

sear_map_data_root = "D:\Datasets\PP-Datasets\RadioMapSeer\\"


def rotate_coordinates(x, y, H, W, k):
    if k == 0:
        return x, y
    elif k == 1:
        return W - 1 - y, x
    elif k == 2:
        return W - 1 - x, H - 1 - y
    elif k == 3:
        return y, H - 1 - x

def data_augmentation(conditions, target, prompt):
    H, W = target.shape[:2]
    k = random.randint(0, 3)
    conditions = np.rot90(conditions, k).copy()
    target = np.rot90(target, k).copy()
    if random.random() < 0.5:
        conditions = np.flip(conditions, axis=1).copy()
        target = np.flip(target, axis=1).copy()
        flipped_h = True
    else:
        flipped_h = False

    if random.random() < 0.5:
        conditions = np.flip(conditions, axis=0).copy()
        target = np.flip(target, axis=0).copy()
        flipped_v = True
    else:
        flipped_v = False

    m = re.search(r'x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)', prompt)

    if m:
        x, y = int(m.group(1)), int(m.group(2))

        x, y = rotate_coordinates(x, y, H, W, k)

        if flipped_h:
            y = H - 1 - y

        if flipped_v:
            x = W - 1 - x
        prompt = re.sub(r'x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)', f'x={x}, y={y}', prompt)
    return conditions, target, prompt

class RandomCutout:
    def __init__(self, num_holes=1, mask_size=16, fill_value=0.0):
        self.num_holes = num_holes
        self.mask_size = mask_size
        self.fill_value = fill_value

    def __call__(self, img):
        """
        Args:
            img: np.ndarray of shape (H, W, C), float32
        Returns:
            np.ndarray of same shape
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)

        _, h, w = img.shape
        for _ in range(self.num_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)

            y1 = max(0, y - self.mask_size // 2)
            y2 = min(h, y + self.mask_size // 2)
            x1 = max(0, x - self.mask_size // 2)
            x2 = min(w, x + self.mask_size // 2)

            img[:, y1:y2, x1:x2] = self.fill_value

        return img.permute(1, 2, 0).numpy().astype(np.float32)


class RadioMapDataset(Dataset):
    def __init__(self, args, data_root="/userhome/data/PP-Datasets/RadioMapPCL/RadioMap2D/", partition='train'):
        self.data = []
        with open(os.path.join(data_root, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.partition = partition
        self.data_root = data_root
        self.channel_in = args.channel_in

        if self.partition == 'train':
            self.data = self.data[:int(len(self.data) / 9 * 8)]
        elif self.partition == 'test':
            self.data = self.data[int(len(self.data) / 9 * 8):]
        else:
            raise ValueError('Invalid partition: {}'.format(self.partition))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # 用cv2读取单通道图片
        if self.channel_in == 1:
            source = cv2.imread(self.data_root + source_filename, -1)
            target = cv2.imread(self.data_root + target_filename, -1)
        else:
            source = cv2.imread(self.data_root + source_filename)
            target = cv2.imread(self.data_root + target_filename)

        # 单通道image的cv2处理
        if len(source.shape) == 2:
            source = np.expand_dims(source, axis=2)
            target = np.expand_dims(target, axis=2)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        import pdb; pdb.set_trace()

        return dict(jpg=target, txt=prompt, hint=source)

class RadioMapSeerDataset_RadioDiff(Dataset):
    def __init__(self, args, data_root="/userhome/zoulk/data/PP-Datasets/RadioMapSeer/", partition='train', with_data_augmentation=True):
        self.partition = partition
        self.training = ((partition == 'train') & with_data_augmentation)

        self.data_root = data_root
        self.channel_in = args.channel_in
        self.carsInput = args.carsInput
        self.simulation = args.simulation

        self.train_maps_inds = np.arange(0, 500, 1, dtype=np.int32)
        self.val_maps_inds = np.arange(500, 600, 1, dtype=np.int32)
        self.test_maps_inds = np.arange(600, 701, 1, dtype=np.int32)

        self.train_data = []
        self.val_data = []
        self.test_data = []

        with open(os.path.join(data_root, f'prompt_{args.prompt_type}/prompt-{args.simulation}.json'), 'rt') as f:
            for line in f:
                file = json.loads(line)
                map_idx = file["source"].split('/')[-1].split('.')[0]
                if int(map_idx) in self.train_maps_inds:
                    self.train_data.append(json.loads(line))
                elif int(map_idx) in self.val_maps_inds:
                    self.val_data.append(json.loads(line))
                elif int(map_idx) in self.test_maps_inds:
                    self.test_data.append(json.loads(line))

        if self.partition == 'train':
            self.data = self.train_data
        elif self.partition == 'val':
            self.data = self.val_data
        elif self.partition == 'test':
            self.data = self.test_data
        else:
            raise ValueError(f'Invalid partition: {self.partition}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        building_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # follow RadioDiff
        antenna_filename = building_filename.replace('buildings_complete', 'antennas')
        image_id = target_filename.split('/')[-1].split('.')[0]
        antenna_id = antenna_filename.split('/')[-1].split('.')[0]
        antenna_filename = antenna_filename.replace(antenna_id, image_id)
        car_filename = building_filename.replace('buildings_complete', 'cars')

        # 用cv2读取单通道图片
        buildings = cv2.imread(os.path.join(self.data_root, building_filename), -1)
        antennas = cv2.imread(os.path.join(self.data_root, antenna_filename), -1)

        if self.carsInput=="no":
            conditions = np.stack([buildings, antennas, buildings], axis=2)
        else:
            target_filename = target_filename.replace(self.simulation, "cars"+self.simulation)
            #### 2026-01-26 modify
            cars = cv2.imread(os.path.join(self.data_root, car_filename), -1)
            # concat buildings, antennas, cars in the last channel
            conditions = np.stack([buildings, antennas, buildings], axis=2)
            cars = cv2.imread(os.path.join(self.data_root, car_filename))
            cars = cv2.cvtColor(cars, cv2.COLOR_BGR2RGB) 
            conditions = conditions + cars
            ####

        # 用cv2读取单通道图片
        if self.channel_in == 1:
            target = cv2.imread(os.path.join(self.data_root, target_filename), -1)
        else:
            target = cv2.imread(os.path.join(self.data_root, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize conditions images to [0, 1].
        conditions = conditions.astype(np.float32) / 255.0
        buildings =  np.stack([buildings, buildings, buildings], axis=2)
        buildings = buildings.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        # data augmentation for training
        if self.training:
            conditions, target, prompt = data_augmentation(conditions, target, prompt)
        return dict(jpg=target, txt=prompt, hint=conditions, path=target_filename, mask=1.0-buildings)

class RadioMapSeerDataset(Dataset):
    def __init__(self, args, data_root="/userhome/zoulk/data/PP-Datasets/RadioMapSeer/", partition='train', with_data_augmentation=True):
        self.partition = partition
        self.training = ((partition == 'train') & with_data_augmentation)

        self.data_root = data_root
        self.channel_in = args.channel_in
        self.carsInput = args.carsInput
        self.simulation = args.simulation

        self.train_maps_inds = np.arange(0, 500, 1, dtype=np.int32)
        self.val_maps_inds = np.arange(500, 600, 1, dtype=np.int32)
        self.test_maps_inds = np.arange(600, 701, 1, dtype=np.int32)

        self.train_data = []
        self.val_data = []
        self.test_data = []

        with open(os.path.join(data_root, f'prompt_{args.prompt_type}/prompt-{args.simulation}.json'), 'rt') as f:
            for line in f:
                file = json.loads(line)
                map_idx = file["source"].split('/')[-1].split('.')[0]
                if int(map_idx) in self.train_maps_inds:
                    self.train_data.append(json.loads(line))
                elif int(map_idx) in self.val_maps_inds:
                    self.val_data.append(json.loads(line))
                elif int(map_idx) in self.test_maps_inds:
                    self.test_data.append(json.loads(line))

        if self.partition == 'train':
            self.data = self.train_data
        elif self.partition == 'val':
            self.data = self.val_data
        elif self.partition == 'test':
            self.data = self.test_data
        else:
            raise ValueError(f'Invalid partition: {self.partition}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        building_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # 用cv2读取单通道图片
        # Do not forget that OpenCV read images in BGR order.
        buildings = cv2.imread(os.path.join(self.data_root, building_filename))
        buildings = cv2.cvtColor(buildings, cv2.COLOR_BGR2RGB)

        target = cv2.imread(os.path.join(self.data_root, target_filename))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.carsInput=="no":
            conditions = buildings
        else:

            car_filename = building_filename.replace('buildings_complete', 'cars')
            target_filename = target_filename.replace(self.simulation, "cars"+self.simulation)

            cars = cv2.imread(os.path.join(self.data_root, car_filename))
            cars = cv2.cvtColor(cars, cv2.COLOR_BGR2RGB)

            conditions = buildings + cars

        # Normalize conditions images to [0, 1].
        conditions = conditions.astype(np.float32) / 255.0
        buildings = buildings.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target / 127.5) - 1.0

        # data augmentation for training
        if self.training:
            conditions, target, prompt = data_augmentation(conditions, target, prompt)
        return dict(jpg=target, txt=prompt, hint=conditions, path=target_filename, mask=1.0-buildings)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--channel_in', type=int, default=3)
    parser.add_argument('--simulation', type=str, default='IRT4',
                        help='simulation type for RadioMapSear dataset: DPM, IRT2, IRT4')
    parser.add_argument('--prompt_type', type=str, default='v6')

    # using cars image or not
    parser.add_argument('--carsInput', type=str, default='yes')

    args = parser.parse_args()
    dataset = RadioMapSeerDataset(args, data_root=sear_map_data_root)
    print(len(dataset))

    item = dataset[1]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']

    print(txt)
    print(jpg.shape)
    print(hint.shape)

    # 遍历 dataset
    for i in range(len(dataset)):
        item = dataset[i]
        import pdb;  pdb.set_trace()
