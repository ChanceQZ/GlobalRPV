import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

import mmcv
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

def auto_pad(image, step_size, windows_size):
    height, width = image.shape[0], image.shape[1]

    h_padding = step_size - (height - windows_size + step_size) % step_size
    w_padding = step_size - (width - windows_size + step_size) % step_size
        
    if h_padding == windows_size and w_padding == windows_size:
        h_padding, w_padding = 0, 0
    
    return h_padding, w_padding

def sliding_crop(image, step_size, windows_size):
    channels = len(image.shape)
    
    h_padding, w_padding = auto_pad(image, step_size, windows_size)

    if channels == 2:
        image_pad = np.pad(image, ((0, h_padding), (0, w_padding)))
    elif channels == 3:
        image_pad = np.pad(image, ((0, h_padding), (0, w_padding), (0, 0)))
        
    for row in range(0, image_pad.shape[0], step_size):
        for col in range(0, image_pad.shape[1], step_size):
            if channels == 3:
                crop_image = image_pad[row:row + windows_size, col:col + windows_size, :]
            elif channels == 2:
                crop_image = image_pad[row:row + windows_size, col:col + windows_size]

            if crop_image.shape[0] == windows_size and crop_image.shape[1] == windows_size:
                yield crop_image

                
class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
    
    
def inference_segmentor(model, imgs):
    """Inference image(s) with the segmentor.
    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = imgs if isinstance(imgs, list) else [imgs]
    for img in imgs:
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def make_grid(pred_list, img, step_size, windows_size):
    h_padding, _ = auto_pad(img, step_size, windows_size)

    n_row = (img.shape[0] - windows_size + h_padding + step_size) // step_size
    n_col = len(pred_list) // n_row
    
    temp = np.zeros((windows_size * n_row, windows_size * n_col))

    n = 0
    for iidx in range(0, windows_size * n_row, windows_size):
        for jidx in range(0, windows_size * n_col, windows_size):
            temp[iidx:iidx+windows_size, jidx:jidx+windows_size] = pred_list[n]
            n += 1
    
    return temp


def pair_plot(image, label):
    image = mmcv.bgr2rgb(image)
    label = np.where(label==1, 255, label).astype(image.dtype)
    if len(label.shape) == 2:
        label = np.repeat(label[:,:,None], 3, axis=2)
    
    overlapping = cv2.addWeighted(image, 0.5, label, 0.5, 0)
    
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(image)
    ax2.imshow(label)
    ax3.imshow(overlapping)
    
    plt.subplots_adjust(wspace=0.01, hspace=0)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    
    
def voting_ensemble(pred_list_1, pred_list_2):
    return [np.logical_and(pred_1, pred_2).astype(np.uint8) \
            for pred_1, pred_2 in zip(pred_list_1, pred_list_2)]


def union_ensemble(pred_list_1, pred_list_2):
    return [np.logical_or(pred_1, pred_2).astype(np.uint8) \
            for pred_1, pred_2 in zip(pred_list_1, pred_list_2)]


def sparse_save(save_path, arr):
    sparse = coo_matrix(arr)
    np.savez(save_path, 
             shape=arr.shape,
             row=sparse.row, 
             col=sparse.col,
             data=sparse.data)