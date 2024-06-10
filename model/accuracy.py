
import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from collections import defaultdict
from data.dataloader import get_dataloader, get_transform
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

def test_accuracy(model,dataloader, save_img=False):
    model.eval()
    TP, FP, FN = 0, 0, 0
    assert dataloader.batch_size == 1
    # if save_img:
    #     save_dir = join(opt.save_dir, 'images')
    #     os.makedirs(save_dir, exist_ok=True)
    #     global_idx = 1
    #     un_norm = get_transform('unnorm', opt, -1)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
        # for batch_idx, data in enumerate(dataloader):
            # load
       
            # forward
           
            expressions = data['target_expressions']
            expressions = expressions[0].split(',')
            labels = data['target_labels'][0]
            # forward
            inputs = dict(
                local_images=data['cropped_images'].cuda().repeat_interleave(len(expressions), dim=0),
                global_image=data['global_images'].cuda().repeat_interleave(len(expressions), dim=0),
                sentences=expressions,
            )
            logits = model(inputs)['scores'].cpu()
            # evaluate
            TP += ((logits >= 0) * (labels == 1)).sum().item()
            FP += ((logits >= 0) * (labels == 0)).sum().item()
            FN += ((logits < 0) * (labels == 1)).sum().item()
            # save images
            # if save_img:
            #     local_img = data['cropped_images'].squeeze(0)
            #     global_img = data['global_images'].squeeze(0)
            #     local_img = F.interpolate(local_img, global_img.size()[2:])
            #     imgs = un_norm(
            #         torch.cat(
            #             (local_img, global_img),
            #             dim=0
            #         )
            #     )
            #     imgs = imgs.repeat(len(expressions), 1, 1, 1, 1)
            #     for i in range(len(imgs)):
            #         file_name = '{}_{}_{:.0f}_{:.2f}.jpg'.format(
            #             global_idx,
            #             expressions[i].replace(' ', '-'),
            #             labels[i],
            #             logits[i]
            #         )
            #         save_image(
            #             imgs[i],
            #             join(save_dir, file_name)
            #         )
            #         global_idx += 1
    if TP == 0:
        PRECISION = 0
        RECALL = 0
    else:
        PRECISION = TP / (TP + FP) * 100
        RECALL = TP / (TP + FN) * 100
    print(TP, FP, FN)
    return PRECISION, RECALL