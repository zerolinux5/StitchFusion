import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from typing import Tuple
from torch.utils.data import DataLoader
from semseg.augmentations_mm import get_train_augmentation
from .labels import train_classes, id2trainId, train_pallet

class KITTI(Dataset):
    """
    num_classes: 19
    """

    # PALETTE = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
    #             [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    # ID2TRAINID = {0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 
    # 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 34:2, 35:4, 36:255, 37:5, 38:255, 39:255, 40:255, 41:255, 42:255, 43:255, 44:255, -1:255}

    def __init__(self, root: str = 'data/KITTI', split: str = 'train', transform = None, modals = ['img', 'depth', 'intensity', 'lidar'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.transform = transform
        self.CLASSES = train_classes
        self.PALETTE = train_pallet
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals

        # self.label_map = np.arange(256)
        # for id, trainid in self.ID2TRAINID.items():
        #     self.label_map[id] = trainid
        file_name = f"{split}_frames.txt"
        self.file_path = os.path.join(root, file_name)
        self.files = open(self.file_path).readlines()
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        file_dir, file_num = self.files[index].strip().split()
        rgb = os.path.join(self.root, "img", file_dir, "image_00", f"{file_num}_rgb.png")
        x1 = os.path.join(self.root, "hha", file_dir, "image_00", f"{file_num}_hha.png")
        x2 = os.path.join(self.root, "lidar", file_dir, "image_00", f"{file_num}_lidar.png")
        x3 = os.path.join(self.root, "intensity", file_dir, "image_00", f"{file_num}_intensity.png")
        lbl_path = os.path.join(self.root, "semantics_fixed", file_dir, "image_00", f"{file_num}.png")

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if 'depth' in self.modals:
            sample['depth'] = self._open_img(x1)
        if 'lidar' in self.modals:
            sample['lidar'] = self._open_img(x2)
        if 'intensity' in self.modals:
            sample['intensity'] = self._open_img(x3)
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        # label = self.label_map[label]
        # fixed_label = np.zeros(label.shape, dtype=np.int32)
        # for key, value in id2trainId.items():
        #     index = np.where(label == key)
        #     fixed_label[index] = value
        return torch.from_numpy(label)


if __name__ == '__main__':
    traintransform = get_train_augmentation((376, 1408), seg_fill=255)

    trainset = KITTI(transform=traintransform)
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=True, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
        print(torch.unique(lbl))