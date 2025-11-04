import os
import glob
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import shutil
from torchvision import io
from semseg.datasets.labels import id2trainId

root = "data"
dataset = "KITTI"
in_root = "semantics"
out_root = "semantics_fixed"
cam = "image_00"

sem_pattern = os.path.join(root, dataset, in_root, '**', cam,'*.png')
sem_images = sorted(glob.glob(sem_pattern, recursive=True))

for image_path in sem_images:
    sequence = image_path.split("/")[3]
    image_name = image_path.split("/")[-1]
    out_path = os.path.relpath(os.path.join(root, dataset, out_root, sequence, cam, image_name))
    label = io.read_image(image_path)[0,...].numpy()
    fixed_label = np.zeros(label.shape, dtype=np.uint8)
    for key, value in id2trainId.items():
        index = np.where(label == key)
        fixed_label[index] = value
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, fixed_label)