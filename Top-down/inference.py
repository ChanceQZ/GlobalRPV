import os
import glob
import cv2
import mmcv
import torch
import numpy as np
from tqdm import tqdm
from mmseg.apis import init_segmentor

from utils import *


config_file = './segformer_mit-b5_512x512_160k_roof.py'
checkpoint_file = ''

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

img_path_list = glob.glob('')
for img_path in tqdm(img_path_list):
    save_path = img_path.replace('', '')
    if os.path.exists(save_path):
        continue
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    img = cv2.imread(img_path)

    step_size, windows_size = 512, 512
    height, width = img.shape[0], img.shape[1]

    img_list = list(sliding_crop(img, step_size, windows_size))
    pred_list = []
    step = 128
    for idx in range(0, len(img_list), step):
        img_chunk = img_list[idx : idx + step]
        
        pred_chunk = inference_segmentor(model, img_chunk)
        
        pred_list.extend(pred_chunk)

    pred_merge = make_grid(pred_list, img, step_size, windows_size)[:height, :width].astype(np.uint8)
    
    
    cv2.imwrite(save_path, pred_merge)