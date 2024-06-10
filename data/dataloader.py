import os
import json
import random
import numpy as np
from PIL import Image
from os.path import join
from numpy.random import choice
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from utils import *
# from opts import opt



VIDEOS = {
    'train': [
        '0001', '0002', '0003', '0004', '0006',
        '0007', '0008', '0009', '0010', '0012',
        '0014', '0015', '0016', '0018', '0020',
    ],
    'val': [
        '0005', '0011', '0013'
    ],
    'test': [
        '0005', '0011', '0013'
    ],
}


RESOLUTION = {
    '0001': (375, 1242), '0002': (375, 1242), '0003': (375, 1242), '0004': (375, 1242),
    '0005': (375, 1242), '0006': (375, 1242), '0007': (375, 1242), '0008': (375, 1242),
    '0009': (375, 1242), '0010': (375, 1242), '0011': (375, 1242), '0012': (375, 1242),
    '0013': (375, 1242), '0014': (370, 1224), '0015': (370, 1224), '0016': (370, 1224),
    '0018': (374, 1238), '0020': (376, 1241),
}


WORDS_MAPPING = {
    'cars': 'car',
    'vehicles': 'car',
    'people': 'pedestrian',
    'persons': 'pedestrian',
    'males': 'men',
    'females': 'women',
    'light-colors': 'light-color',
}


WORDS = {
    'dropped': [
        'in', 'the', 'with', 'direction', 'of',
        'ours', 'camera', 'and', 'which', 'are',
        'than', 'carrying', 'holding', 'a', 'bag',
        'pants', 'who', 'horizon',
    ],
    'category': ['car', 'pedestrian', 'men', 'women'],
    'color': ['black', 'red', 'silver', 'light-color', 'white'],
    'location': ['right', 'left', 'front'],
    'direction': ['same', 'counter'],
    'status': ['moving', 'turning', 'parking', 'braking', 'walking', 'standing'],
    'speed': ['faster', 'slower'],
}


EXPRESSIONS = {
    'train': [
        'black-cars', 'black-cars-in-right', 'black-cars-in-the-left', 'black-cars-with-the-counter-direction-of-ours',
        'black-moving-cars', 'black-moving-vehicles', 'black-vehicles', 'black-vehicles-in-right',
        'black-vehicles-in-the-left', 'black-vehicles-with-the-counter-direction-of-ours', 'cars', 'cars-in-black',
        'cars-in-front-of-ours', 'cars-in-front-of-the-camera', 'cars-in-horizon-direction', 'cars-in-left',
        'cars-in-light-color', 'cars-in-red', 'cars-in-right', 'cars-in-silver', 'cars-in-the-counter-direction',
        'cars-in-the-counter-direction-of-ours', 'cars-in-the-left', 'cars-in-the-same-direction-of-ours',
        'cars-which-are-braking', 'cars-which-are-in-the-left-and-turning', 'cars-which-are-parking',
        'cars-which-are-slower-than-ours', 'cars-which-are-turning', 'cars-with-the-counter-direction-of-ours',
        'cars-with-the-same-direction-of-ours', 'counter-direction-cars-in-the-left',
        'counter-direction-cars-in-the-right', 'counter-direction-vehicles-in-the-left',
        'counter-direction-vehicles-in-the-right', 'left-cars', 'left-cars-in-black', 'left-cars-in-light-color',
        'left-cars-in-red', 'left-cars-in-silver', 'left-cars-in-the-counter-direction-of-ours',
        'left-cars-in-the-same-direction-of-ours', 'left-cars-which-are-black', 'left-cars-which-are-in-light-colors',
        'left-cars-which-are-parking', 'left-moving-cars', 'left-moving-vehicles', 'left-vehicles',
        'left-vehicles-in-black', 'left-vehicles-in-light-color', 'left-vehicles-in-red', 'left-vehicles-in-silver',
        'left-vehicles-in-the-counter-direction-of-ours', 'left-vehicles-in-the-same-direction-of-ours',
        'left-vehicles-which-are-black', 'left-vehicles-which-are-in-light-colors', 'left-vehicles-which-are-parking',
        'light-color-cars', 'light-color-cars-in-the-left', 'light-color-cars-in-the-right',
        'light-color-cars-which-are-parking', 'light-color-cars-with-the-counter-direction-of-ours',
        'light-color-moving-cars', 'light-color-moving-vehicles', 'light-color-vehicles',
        'light-color-vehicles-in-the-left', 'light-color-vehicles-in-the-right',
        'light-color-vehicles-which-are-parking', 'light-color-vehicles-with-the-counter-direction-of-ours',
        'males-in-the-left', 'males-in-the-right', 'men-in-the-left', 'men-in-the-right', 'moving-black-cars',
        'moving-black-vehicles', 'moving-cars', 'moving-cars-in-the-same-direction-of-ours', 'moving-turning-cars',
        'moving-turning-vehicles', 'moving-vehicles', 'moving-vehicles-in-the-same-direction-of-ours', 'parking-cars',
        'parking-cars-in-the-left', 'parking-cars-in-the-right', 'parking-vehicles', 'parking-vehicles-in-the-left',
        'parking-vehicles-in-the-right', 'pedestrian', 'pedestrian-in-the-left', 'pedestrian-in-the-pants',
        'pedestrian-in-the-right', 'people', 'people-in-the-left', 'people-in-the-pants', 'people-in-the-right',
        'persons', 'persons-in-the-left', 'persons-in-the-pants', 'persons-in-the-right', 'red-cars-in-the-left',
        'red-cars-in-the-right', 'red-moving-cars', 'red-moving-vehicles', 'red-turning-cars', 'red-turning-vehicles',
        'red-vehicles-in-the-left', 'red-vehicles-in-the-right', 'right-cars-in-black', 'right-cars-in-light-color',
        'right-cars-in-red', 'right-cars-in-silver', 'right-cars-in-the-counter-direction-of-ours',
        'right-cars-in-the-same-direction-of-ours', 'right-cars-which-are-black',
        'right-cars-which-are-in-light-colors', 'right-cars-which-are-parking', 'right-moving-cars',
        'right-moving-vehicles', 'right-vehicles-in-black', 'right-vehicles-in-light-color', 'right-vehicles-in-red',
        'right-vehicles-in-silver', 'right-vehicles-in-the-counter-direction-of-ours',
        'right-vehicles-in-the-same-direction-of-ours', 'right-vehicles-which-are-black',
        'right-vehicles-which-are-in-light-colors', 'right-vehicles-which-are-parking',
        'same-direction-cars-in-the-left', 'same-direction-cars-in-the-right', 'same-direction-vehicles-in-the-left',
        'same-direction-vehicles-in-the-right', 'silver-cars-in-right', 'silver-cars-in-the-left',
        'silver-turning-cars', 'silver-turning-vehicles', 'silver-vehicles-in-right', 'silver-vehicles-in-the-left',
        'turning-cars', 'turning-vehicles', 'vehicles', 'vehicles-in-black', 'vehicles-in-front-of-ours',
        'vehicles-in-front-of-the-camera', 'vehicles-in-horizon-direction', 'vehicles-in-left',
        'vehicles-in-light-color', 'vehicles-in-red', 'vehicles-in-right', 'vehicles-in-silver',
        'vehicles-in-the-counter-direction', 'vehicles-in-the-counter-direction-of-ours', 'vehicles-in-the-left',
        'vehicles-in-the-same-direction-of-ours', 'vehicles-which-are-braking',
        'vehicles-which-are-in-the-left-and-turning', 'vehicles-which-are-parking',
        'vehicles-which-are-slower-than-ours', 'vehicles-which-are-turning',
        'vehicles-with-the-counter-direction-of-ours', 'vehicles-with-the-same-direction-of-ours',
        'walking-pedestrian-in-the-left', 'walking-pedestrian-in-the-right', 'walking-people-in-the-left',
        'walking-people-in-the-right', 'walking-persons-in-the-left', 'walking-persons-in-the-right'],
    'test': [
        'black-cars-in-right', 'black-cars-in-the-left', 'black-vehicles-in-right', 'black-vehicles-in-the-left',
        'cars-in-black', 'cars-in-front-of-ours', 'cars-in-left', 'cars-in-light-color', 'cars-in-right',
        'cars-in-silver', 'cars-in-the-counter-direction-of-ours', 'cars-in-the-left', 'cars-in-the-right',
        'cars-in-the-same-direction-of-ours', 'cars-in-white', 'cars-which-are-faster-than-ours',
        'counter-direction-cars-in-the-left', 'counter-direction-vehicles-in-the-left', 'females',
        'females-in-the-left', 'females-in-the-right', 'left-cars-in-black', 'left-cars-in-light-color',
        'left-cars-in-silver', 'left-cars-in-the-counter-direction-of-ours',
        'left-cars-in-the-same-direction-of-ours', 'left-cars-in-white', 'left-cars-which-are-parking',
        'left-pedestrian-who-are-walking', 'left-people-who-are-walking', 'left-persons-who-are-walking',
        'left-vehicles-in-black', 'left-vehicles-in-light-color', 'left-vehicles-in-silver',
        'left-vehicles-in-the-counter-direction-of-ours', 'left-vehicles-in-the-same-direction-of-ours',
        'left-vehicles-in-white', 'left-vehicles-which-are-parking', 'light-color-cars-in-the-left',
        'light-color-cars-in-the-right', 'light-color-vehicles-in-the-left', 'light-color-vehicles-in-the-right',
        'males', 'males-in-the-left', 'males-in-the-right', 'men', 'men-in-the-left', 'men-in-the-right',
        'moving-cars', 'moving-left-pedestrian', 'moving-pedestrian', 'moving-right-pedestrian', 'moving-vehicles',
        'parking-cars', 'parking-vehicles', 'pedestrian', 'pedestrian-in-the-left', 'pedestrian-in-the-right',
        'pedestrian-who-are-walking', 'people', 'people-in-the-left', 'people-in-the-right',
        'people-who-are-walking', 'persons', 'persons-in-the-left', 'persons-in-the-right',
        'persons-who-are-walking', 'right-cars-in-black', 'right-cars-in-light-color', 'right-cars-in-silver',
        'right-cars-in-white', 'right-cars-which-are-parking', 'right-pedestrian-who-are-walking',
        'right-people-who-are-walking', 'right-persons-who-are-walking', 'right-vehicles-in-black',
        'right-vehicles-in-light-color', 'right-vehicles-in-silver', 'right-vehicles-in-white',
        'right-vehicles-which-are-parking', 'same-direction-cars-in-the-left',
        'same-direction-vehicles-in-the-left', 'silver-cars-in-right', 'silver-cars-in-the-left',
        'silver-vehicles-in-right', 'silver-vehicles-in-the-left', 'standing-females', 'standing-males',
        'standing-men', 'standing-women', 'turning-cars', 'turning-vehicles', 'vehicles-in-black',
        'vehicles-in-front-of-ours', 'vehicles-in-left', 'vehicles-in-light-color', 'vehicles-in-right',
        'vehicles-in-silver', 'vehicles-in-the-counter-direction-of-ours', 'vehicles-in-the-left',
        'vehicles-in-the-right', 'vehicles-in-the-same-direction-of-ours', 'vehicles-in-white',
        'vehicles-which-are-faster-than-ours', 'walking-females', 'walking-males', 'walking-men',
        'walking-pedestrian', 'walking-women', 'white-cars-in-the-left', 'white-cars-in-the-right',
        'white-vehicles-in-the-left', 'white-vehicles-in-the-right', 'women', 'women-carrying-a-bag',
        'women-holding-a-bag', 'women-in-the-left', 'women-in-the-right'
    ],
    'dropped': [
        'women-back-to-the-camera', 'vehicles-which-are-braking', 'men-back-to-the-camera',
        'vehicles-in-horizon-direction', 'cars-which-are-braking', 'cars-in-horizon-direction',
        'males-back-to-the-camera', 'females-back-to-the-camera',
    ],  # these expressions are not evaluated as in TransRMOT
    '0005': [
        'left-cars-in-silver', 'same-direction-cars-in-the-left', 'counter-direction-cars-in-the-left',
        'left-vehicles-in-the-counter-direction-of-ours', 'silver-vehicles-in-the-left',
        'vehicles-in-front-of-ours', 'right-cars-in-light-color', 'counter-direction-vehicles-in-the-left',
        'same-direction-vehicles-in-the-left', 'light-color-cars-in-the-right', 'light-color-vehicles-in-the-right',
        'left-cars-in-the-counter-direction-of-ours', 'left-vehicles-in-light-color', 'vehicles-in-left',
        'left-vehicles-in-black', 'left-cars-in-black', 'cars-in-the-same-direction-of-ours',
        'left-cars-which-are-parking', 'cars-in-front-of-ours', 'cars-in-light-color', 'moving-vehicles',
        'vehicles-in-the-counter-direction-of-ours', 'cars-which-are-braking', 'left-vehicles-which-are-parking',
        'vehicles-which-are-braking', 'silver-vehicles-in-right', 'vehicles-in-right', 'vehicles-in-silver',
        'left-cars-in-light-color', 'left-vehicles-in-the-same-direction-of-ours', 'vehicles-in-black',
        'black-cars-in-the-left', 'right-cars-in-silver', 'black-vehicles-in-the-left', 'right-vehicles-in-silver',
        'cars-in-left', 'left-cars-in-the-same-direction-of-ours', 'right-vehicles-in-light-color', 'cars-in-black',
        'cars-in-silver', 'moving-cars', 'cars-in-the-counter-direction-of-ours',
        'vehicles-in-the-same-direction-of-ours', 'vehicles-in-light-color', 'cars-in-right',
        'silver-cars-in-right', 'light-color-cars-in-the-left', 'silver-cars-in-the-left',
        'left-vehicles-in-silver', 'light-color-vehicles-in-the-left'
    ],
    '0011': [
        'right-cars-in-black', 'black-vehicles-in-right', 'parking-vehicles', 'left-cars-in-white',
        'white-cars-in-the-right', 'counter-direction-cars-in-the-left',
        'left-vehicles-in-the-counter-direction-of-ours', 'right-cars-in-light-color',
        'counter-direction-vehicles-in-the-left', 'right-cars-which-are-parking',
        'light-color-cars-in-the-right', 'light-color-vehicles-in-the-right',
        'left-cars-in-the-counter-direction-of-ours', 'left-vehicles-in-light-color', 'pedestrian',
        'vehicles-in-left', 'white-vehicles-in-the-left', 'left-vehicles-in-black', 'moving-pedestrian',
        'vehicles-which-are-faster-than-ours', 'left-cars-in-black', 'cars-in-the-same-direction-of-ours',
        'persons-who-are-walking', 'left-cars-which-are-parking', 'white-vehicles-in-the-right',
        'turning-vehicles', 'parking-cars', 'left-vehicles-in-white', 'cars-in-light-color',
        'moving-vehicles', 'vehicles-in-the-counter-direction-of-ours', 'cars-in-horizon-direction',
        'left-vehicles-which-are-parking', 'black-cars-in-right', 'people-who-are-walking',
        'walking-pedestrian', 'vehicles-in-right', 'vehicles-in-white', 'turning-cars', 'right-cars-in-white',
        'right-vehicles-in-black', 'left-cars-in-light-color', 'vehicles-in-black', 'black-cars-in-the-left',
        'black-vehicles-in-the-left', 'right-vehicles-which-are-parking', 'cars-in-left',
        'vehicles-in-horizon-direction', 'right-vehicles-in-light-color', 'cars-which-are-faster-than-ours',
        'cars-in-black', 'people', 'cars-in-white', 'moving-cars', 'pedestrian-who-are-walking', 'persons',
        'cars-in-the-counter-direction-of-ours', 'vehicles-in-the-same-direction-of-ours',
        'vehicles-in-light-color', 'cars-in-right', 'right-vehicles-in-white', 'light-color-cars-in-the-left',
        'white-cars-in-the-left', 'light-color-vehicles-in-the-left'],
    '0013': [
        'walking-males', 'women-back-to-the-camera', 'walking-women', 'women', 'females',
        'left-people-who-are-walking', 'men-back-to-the-camera', 'right-people-who-are-walking', 'males',
        'people-in-the-left', 'persons-in-the-right', 'people-in-the-right', 'left-persons-who-are-walking',
        'females-back-to-the-camera', 'standing-males', 'men-in-the-left', 'women-carrying-a-bag',
        'women-in-the-left', 'men-in-the-right', 'standing-men', 'standing-women', 'vehicles-in-the-right',
        'moving-left-pedestrian', 'women-in-the-right', 'right-persons-who-are-walking', 'cars-in-the-right',
        'men', 'persons-in-the-left', 'males-in-the-left', 'pedestrian-in-the-left',
        'right-pedestrian-who-are-walking', 'women-holding-a-bag', 'males-back-to-the-camera',
        'vehicles-in-the-left', 'walking-men', 'standing-females', 'left-pedestrian-who-are-walking',
        'females-in-the-left', 'pedestrian-in-the-right', 'cars-in-the-left', 'moving-right-pedestrian',
        'walking-females', 'females-in-the-right', 'males-in-the-right'
    ],
}


ID2EXP = {
      0: 'left car which are parking',
      1: 'car in right',
      2: 'right car which are black',
      3: 'left car which are in light-color',
      4: 'men in the left',
      5: 'right car in light-color',
      6: 'right moving car',
      7: 'left car in black',
      8: 'car in front of the camera',
      9: 'right car which are in light-color',
      10: 'counter direction car in the right',
      11: 'same direction car in the right',
      12: 'pedestrian',
      13: 'counter direction car in the left',
      14: 'left moving car',
      15: 'black moving car',
      16: 'red moving car',
      17: 'light-color car in the left',
      18: 'light-color car with the counter direction of ours',
      19: 'silver car in right',
      20: 'left car in red',
      21: 'car in the same direction of ours',
      22: 'same direction car in the left',
      23: 'moving car in the same direction of ours',
      24: 'men in the right',
      25: 'right car in black',
      26: 'car in silver',
      27: 'black car',
      28: 'silver turning car',
      29: 'moving black car',
      30: 'left car in the same direction of ours',
      31: 'car with the counter direction of ours',
      32: 'car in horizon direction',
      33: 'black car in the left',
      34: 'parking car',
      35: 'car which are braking',
      36: 'right car in the counter direction of ours',
      37: 'red car in the right',
      38: 'right car which are parking',
      39: 'walking pedestrian in the right',
      40: 'moving turning car',
      41: 'left car',
      42: 'car with the same direction of ours',
      43: 'left car in silver',
      44: 'light-color car in the right',
      45: 'car in the counter direction',
      46: 'car in the counter direction of ours',
      47: 'silver car in the left',
      48: 'walking pedestrian in the left',
      49: 'black car with the counter direction of ours',
      50: 'car in light-color',
      51: 'left car in light-color',
      52: 'pedestrian in the right',
      53: 'car in front of ours',
      54: 'red turning car',
      55: 'left car in the counter direction of ours',
      56: 'car which are in the left and turning',
      57: 'car in the left',
      58: 'right car in silver',
      59: 'car in left',
      60: 'turning car',
      61: 'light-color car',
      62: 'light-color moving car',
      63: 'moving car',
      64: 'car which are turning',
      65: 'car',
      66: 'car in red',
      67: 'car in black',
      68: 'car which are parking',
      69: 'right car in the same direction of ours',
      70: 'parking car in the left',
      71: 'parking car in the right',
      72: 'car which are slower than ours',
      73: 'pedestrian in the pants',
      74: 'light-color car which are parking',
      75: 'black car in right',
      76: 'right car in red',
      77: 'pedestrian in the left',
      78: 'left car which are black',
      79: 'red car in the left'
 }


EXP_TO_POS_BBOX_NUMS = {
     'black car': 114,
     'black car in right': 1347,
     'black car in the left': 3758,
     'black car with the counter direction of ours': 104,
     'black moving car': 114,
     'car': 1456,
     'car in black': 6358,
     'car in front of ours': 1831,
     'car in front of the camera': 66,
     'car in horizon direction': 897,
     'car in left': 9304,
     'car in light-color': 6489,
     'car in red': 478,
     'car in right': 4877,
     'car in silver': 1841,
     'car in the counter direction': 72,
     'car in the counter direction of ours': 4843,
     'car in the left': 834,
     'car in the same direction of ours': 10230,
     'car which are braking': 839,
     'car which are in the left and turning': 52,
     'car which are parking': 647,
     'car which are slower than ours': 4138,
     'car which are turning': 289,
     'car with the counter direction of ours': 130,
     'car with the same direction of ours': 217,
     'counter direction car in the left': 4087,
     'counter direction car in the right': 368,
     'left car': 72,
     'left car in black': 3706,
     'left car in light-color': 3053,
     'left car in red': 240,
     'left car in silver': 817,
     'left car in the counter direction of ours': 4087,
     'left car in the same direction of ours': 4790,
     'left car which are black': 112,
     'left car which are in light-color': 112,
     'left car which are parking': 3987,
     'left moving car': 165,
     'light-color car': 223,
     'light-color car in the left': 3053,
     'light-color car in the right': 1464,
     'light-color car which are parking': 589,
     'light-color car with the counter direction of ours': 186,
     'light-color moving car': 223,
     'men in the left': 1827,
     'men in the right': 0,
     'moving black car': 217,
     'moving car': 8368,
     'moving car in the same direction of ours': 358,
     'moving turning car': 104,
     'parking car': 7877,
     'parking car in the left': 557,
     'parking car in the right': 61,
     'pedestrian': 115,
     'pedestrian in the left': 2419,
     'pedestrian in the pants': 1984,
     'pedestrian in the right': 253,
     'red car in the left': 240,
     'red car in the right': 187,
     'red moving car': 223,
     'red turning car': 223,
     'right car in black': 1347,
     'right car in light-color': 1464,
     'right car in red': 187,
     'right car in silver': 227,
     'right car in the counter direction of ours': 368,
     'right car in the same direction of ours': 2632,
     'right car which are black': 68,
     'right car which are in light-color': 68,
     'right car which are parking': 4089,
     'right moving car': 526,
     'same direction car in the left': 4790,
     'same direction car in the right': 2632,
     'silver car in right': 227,
     'silver car in the left': 817,
     'silver turning car': 66,
     'turning car': 388,
     'walking pedestrian in the left': 0,
     'walking pedestrian in the right': 0
}


ID_TO_POS_BBOX_NUMS = {
    idx: EXP_TO_POS_BBOX_NUMS[exp] for idx, exp in ID2EXP.items()
}


FRAMES = {
    '0005': (0, 296),
    '0011': (0, 372),
    '0013': (0, 339),
}  # 视频起止帧



class SquarePad:
    """Reference:
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
    """
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


def get_dataloader(mode, opt, dataset='RMOT_Dataset', show=False, **kwargs):
    dataset = eval(dataset)(mode, opt, **kwargs)
    if show:
        dataset.show_information()
    if mode == 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=opt["train_bs"],
            shuffle=True,
            drop_last=True,
            num_workers=opt["num_workers"],
        )
    elif mode == 'test':
        dataloader = DataLoader(
            dataset,
            batch_size=opt["test_bs"],
            shuffle=False,
            drop_last=False,
            num_workers=opt["num_workers"],
        )
    return dataloader


def get_transform(mode, opt, idx):
    if mode == 'train':
        return T.Compose([
            SquarePad(),
            T.RandomResizedCrop(
                opt["img_hw"][idx],
                ratio=opt["random_crop_ratio"]
            ),
            T.ToTensor(),
            # T.Normalize(opt["norm_mean"], opt["norm_std"]),
        ])
    elif mode == 'test':
        return T.Compose([
            SquarePad(),
            T.Resize(opt["img_hw"][idx]),
            T.ToTensor(),
            # T.Normalize(opt["norm_mean"], opt['norm_std']),
        ])
    elif mode == 'unnorm':
        mean = opt["norm_mean"]
        std = opt['norm_std']
        return T.Normalize(
            [-mean[i]/std[i] for i in range(3)],
            [1/std[i] for i in range(3)],
        )


def filter_target_expressions(gt, target_expressions, exp_key, only_car):
    """
    给定“帧级标签”和“视频级exp"，得到帧级exps和对应labels
    """
    OUT_EXPS, OUT_LABELS = list(), list()
    GT_EXPRESSIONS = gt[exp_key]
    for tgt_exp in target_expressions:
        if only_car and ('car' not in tgt_exp):
            continue
        OUT_EXPS.append(tgt_exp)
        if tgt_exp in GT_EXPRESSIONS:
            OUT_LABELS.append(1)
        else:
            OUT_LABELS.append(0)
    return OUT_EXPS, OUT_LABELS


def filter_gt_expressions(gt_expressions, KEY=None):
    OUT_EXPS = list()
    for gt_exp in gt_expressions:
        if KEY is None:
            OUT_EXPS.append(gt_exp)
        else:
            for key in WORDS[KEY]:
                if key in gt_exp:
                    OUT_EXPS.append(gt_exp)
                    break
    return OUT_EXPS


def dd():
    return defaultdict(list)
def ddd():
    return defaultdict(dd)

def multi_dim_dict(n, types):
   
    return defaultdict(ddd)


def expression_conversion(expression):
    """expression => expression_new"""
    expression = expression.replace('-', ' ').replace('light color', 'light-color')
    words = expression.split(' ')
    expression_converted = ''
    for word in words:
        if word in WORDS_MAPPING:
            word = WORDS_MAPPING[word]
        expression_converted += f'{word} '
    expression_converted = expression_converted[:-1]
    return expression_converted

class RMOT_Dataset(Dataset):
    """
    For the `car` + `color+direction+location` settings
    For the `car` + 'status' settings
    """
    def __init__(self, mode, opt, only_car=False):
        super().__init__()
        assert mode in ('train', 'test')
        self.opt = opt
        self.mode = mode
        self.only_car = only_car  # 选择类别
        self.transform = {idx: get_transform(mode, self.opt, idx) for idx in (0, 1, 2)}
        self.exp_key = 'expression_new'  # 经处理后的expression标签
        self.data = self._parse_data()
        self.data_keys = list(self.data.keys())
        self.exp2id = {exp: idx for idx, exp in ID2EXP.items()}

    def _parse_data(self):
        labels = json.load(open(self.opt["rf_kitti_json"]))
        data = multi_dim_dict(2, list)
        target_expressions = defaultdict(list)
        expression_dir = self.opt["rf_expression"]
        for video in VIDEOS[self.mode]:
            # load expressions
            for exp_file in os.listdir(join(expression_dir, video)):
                expression = exp_file.replace('.json', '')
                expression_new = expression_conversion(expression)
                if expression_new not in target_expressions[video]:
                    target_expressions[video].append(expression_new)
            # load data
            H, W = RESOLUTION[video]
            for obj_id, obj_label in labels[video].items():
                num = 0
                for value in obj_label.values():
                    if len(value['category']) > 0 \
                        and (
                            (self.only_car and (value['category'][0] == 'car'))
                            or (not self.only_car)
                        ):
                                num += 1
                if num <= self.opt["sample_frame_len"]:
                    continue
                if len(obj_label) <= self.opt["sample_frame_len"]:
                    continue
                obj_key = f'{video}_{obj_id}'
                pre_frame_id = -1
                curr_data = defaultdict(list)
                for frame_id, frame_label in obj_label.items():
                    # check that the `frame_id` is in order
                    frame_id = int(frame_id)
                    assert frame_id > pre_frame_id
                    pre_frame_id = frame_id
                    # get target exps
                    tgt_exps, tgt_labels = filter_target_expressions(
                        frame_label, target_expressions[video], self.exp_key, self.only_car
                    )
                    if len(tgt_exps) == 0:
                        continue
                    # load exp
                    exps = frame_label[self.exp_key]
                    exps = filter_gt_expressions(exps, None)
                    if len(exps) == 0:
                        continue
                    # load box
                    x, y, w, h = frame_label['bbox']
                    # save
                    curr_data['expression'].append(exps)
                    curr_data['target_expression'].append(tgt_exps)
                    curr_data['target_labels'].append(tgt_labels)
                    curr_data['bbox'].append([frame_id, x * W, y * H, (x + w) * W, (y + h) * H])
                if len(curr_data['bbox']) > self.opt["sample_frame_len"]:
                    data[obj_key] = curr_data.copy()
        return data

    def _crop_image(self, images, indices, data, mode):
        if mode == 'small':
            crops = torch.stack(
                [self.transform[0](
                    images[i].crop(data['bbox'][idx][1:])
                ) for i, idx in enumerate(indices)],
                dim=0
            )
        elif mode == 'big':
            X1, Y1, X2, Y2 = 1e5, 1e5, -1, -1
            for idx in indices:
                x1, y1, x2, y2 = data['bbox'][idx][1:]
                X1, Y1, X2, Y2 = min(X1, x1), min(Y1, y1), max(X2, x2), max(Y2, y2)
            crops = torch.stack(
                [self.transform[0](
                    image.crop([X1, Y1, X2, Y2])
                ) for image in images],
                dim=0
            )
        return crops

    def __getitem__(self, index):
        data_key = self.data_keys[index]
        video = data_key.split('_')[0]
        data = self.data[data_key]

        # sample frames
        data_len = len(data['bbox'])
        sample_len = self.opt["sample_frame_len"]
        sample_num = self.opt["sample_frame_num"]
        sampled_indices = list()
        if self.mode == 'train':
            # continuous random sampling
            start_idx = random.randint(0, data_len - sample_len)
            stop_idx = start_idx + sample_len
            # restricted random sampling
            step = sample_len // sample_num
            for idx in range(start_idx, stop_idx, step):
                sampled_indices.append(
                    random.randint(idx, idx + step - 1)
                )
        elif self.mode == 'test':
            # continuous sampling
            start_idx = index % (data_len - sample_len)
            stop_idx = start_idx + sample_len
            # restricted sampling
            step = sample_len // sample_num
            for idx in range(start_idx, stop_idx, step):
                sampled_indices.append(idx + step // 2)

        # load images
        images = [
            Image.open(
                join(
                    self.opt["data_root"],
                    'KITTI/training/image_02/{}/{:0>6d}.png'
                        .format(video, data['bbox'][idx][0])
                )
            ) for idx in sampled_indices
        ]

        # load expressions
        expressions = list()
        for idx in sampled_indices:
            expressions.extend(data['expression'][idx])
        expressions = sorted(list(set(expressions)))

        # crop images
        cropped_images = self._crop_image(
            images, sampled_indices, data, 'small'
        )  # [T,C,H,W]

        # global images
        global_images = torch.stack([
            self.transform[2](image)
            for image in images
        ], dim=0)

        # sample target expressions
        if self.mode == 'train':
            idx = choice(sampled_indices, size=1)[0]
        elif self.mode == 'test':
            idx = sampled_indices[len(sampled_indices) // 2]
        target_expressions = data['target_expression'][idx]
        target_labels = data['target_labels'][idx]
        if self.mode == 'train':
            assert self.opt["sample_expression_num"] == 1
            sampled_target_idx = choice(
                range(len(target_expressions)),
                size=1,
                replace=False
            )
            sampled_target_exp = [
                target_expressions[i]
                for i in sampled_target_idx
            ]
            sampled_target_label = [
                target_labels[i]
                for i in sampled_target_idx
            ]
            exp_id = self.exp2id[sampled_target_exp[0]]
        elif self.mode == 'test':
            sampled_target_exp = target_expressions
            sampled_target_label = target_labels
            exp_id = -1

        sampled_target_label = torch.tensor(
            sampled_target_label,
            dtype=float
        )
        return dict(
            cropped_images=cropped_images,
            global_images=global_images,
            expressions=','.join(expressions),
            target_expressions=','.join(sampled_target_exp),
            target_labels=sampled_target_label,
            expression_id=exp_id,
            start_idx=start_idx,
            stop_idx=stop_idx,
            data_key=data_key,
        )

    def __len__(self):
        return len(self.data_keys)

    def show_information(self):
        print(
            f'===> Refer-KITTI ({self.mode}) <===\n'
            f"Number of identities: {len(self.data)}"
        )


class Track_Dataset(Dataset):
    def __init__(self, mode, opt):
        self.opt = opt
        self.mode = mode
        self.transform = {idx: get_transform(self.mode, self.opt, idx) for idx in (0, 1, 2)}
        self.data = self._parse_data()

    def _parse_data(self):
        sample_length = self.opt["sample_frame_len"]
        sample_stride = self.opt["sample_frame_stride"]
        DATA = list()
        for video in VIDEOS[self.mode]:
            # load tracks
            tracks_1 = np.loadtxt(join(self.opt["track_root"], video, 'car', 'predict.txt'), delimiter=',')
            if len(tracks_1.shape) == 2:
                tracks = tracks_1
                max_obj_id = max(tracks_1[:, 1])
            else:
                tracks = np.empty((0, 10))
                max_obj_id = 0
            tracks_2 = np.loadtxt(join(self.opt["track_root"], video, 'pedestrian', 'predict.txt'), delimiter=',')
            if len(tracks_2.shape) == 2:
                tracks_2[:, 1] += max_obj_id
                tracks = np.concatenate((tracks, tracks_2), axis=0)
            tracks = tracks[np.lexsort([tracks[:, 0], tracks[:, 1]])]  # ID->frame
            # parse tracks
            ids = set(tracks[:, 1])
            for obj_id in ids:
                tracks_id = tracks[tracks[:, 1] == obj_id]
                frame_min, frame_max = int(min(tracks_id[:, 0])), int(max(tracks_id[:, 0]))
                # 识别轨迹断点位置，从而方便对每个sub-tracklet单独处理
                frame_pairs, start_frame, stop_frame = list(), frame_min, -1
                previous_frame = start_frame - 1
                for frame_idx in list(tracks_id[:, 0]) + [1e5]:
                    if frame_idx != previous_frame + 1:
                        stop_frame = previous_frame
                        frame_pairs.append([int(start_frame), int(stop_frame)])
                        start_frame = frame_idx
                    previous_frame = frame_idx
                # 将tracklets按sample_stride划分为片段
                total_length = 0
                for f_min, f_max in frame_pairs:
                    total_length += (f_max - f_min + 1)
                    for f_idx in range(f_min, f_max + 1, sample_stride):
                        f_stop = min(f_max, f_idx + sample_length - 1)
                        f_start = max(f_min, f_stop - sample_length + 1)
                        tracklets = tracks_id[np.isin(
                            tracks_id[:, 0],
                            range(f_start, f_stop + 1)
                        )][:, :6]
                        tracklets[:, 4:6] += tracklets[:, 2:4]
                        tracklets = tracklets.astype(int)
                        assert (f_stop - f_start + 1) == len(tracklets)
                        for expression in EXPRESSIONS[video]:
                            DATA.append(dict(
                                video=video,
                                obj_id=int(obj_id),
                                start_frame=f_start,
                                stop_frame=f_stop,
                                tracklets=tracklets,
                                expression=expression,
                            ))
                        if f_stop == f_max:
                            break
                assert total_length == len(tracks_id)
        return DATA

    def __getitem__(self, index):
        video, obj_id, start_frame, stop_frame, tracklets, expression = self.data[index].values()
        assert (stop_frame - start_frame + 1) == len(tracklets)

        # expression conversion
        expression_converted = expression_conversion(expression)

        # frame sampling
        sampled_indices = np.linspace(
            0, len(tracklets),
            self.opt["sample_frame_num"],
            endpoint=False, dtype=int
        )
        sampled_tracklets = tracklets[sampled_indices]

        # load images
        images = [
            Image.open(
                join(
                    self.opt["data_root"],
                    'KITTI/training/image_02/{}/{:0>6d}.png'
                        .format(video, bbox[0])
                )
            ) for bbox in sampled_tracklets
        ]

        # crop images
        cropped_images = torch.stack(
            [self.transform[0](
                images[i].crop(bbox[2:6])
            ) for i, bbox in enumerate(sampled_tracklets)],
            dim=0
        )

        # global images
        global_images = torch.stack([
            self.transform[2](image)
            for image in images
        ], dim=0)

        return dict(
            video=video,
            obj_id=obj_id,
            start_frame=start_frame,
            stop_frame=stop_frame,
            cropped_images=cropped_images,
            global_images=global_images,
            expression_raw=expression,
            expression_new=expression_converted,
        )

    def __len__(self):
        return len(self.data)
