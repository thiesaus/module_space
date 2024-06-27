# @Author       : Ruopeng Gao
# @Date         : 2022/12/2
import math
import os
import torch
import random
from numpy.random import choice

from collections import defaultdict
from random import randint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import cv2
import torchvision.transforms as T

import json
import os
from os import listdir,walk
from os.path import isfile, join

from collections import defaultdict

def dd():
    return defaultdict(list)
def ddd():
    return defaultdict(dd)
def dddd():
    return defaultdict(ddd)
def ddddd():
    return defaultdict(dddd)
def dddddd():
    return defaultdict(ddddd)


def standardlize(json_path,data_root):
    overall= dict()
    for i in ['train']:
        temp_path=os.path.join(data_root,i,'BDD')
        array=[f.split('.')[0] for f in listdir(temp_path) if isfile(join(temp_path, f)) and f.endswith('.mov')]
        overall[i]=dict([('videos',array)])

    with open(json_path, 'r') as f:
        data_pool = json.load(f)

    for j in ['train']:
        new_list=[]
        for i in data_pool['videos']:
            if i['metadata']['dataset'] == 'BDD' and i['name'].split('/')[2] in overall[j]['videos']:
                i['video_name']=i['name'].split('/')[-1]
                new_list.append(i)
        overall[j]['videos']=new_list
    
    video_dict=dict()
    for i in ['train']:
        video_ids=[]
        for j in overall[i]['videos']:
            video_ids.append(j['id'])
        video_dict[i]=video_ids

    annotation_dict= dict()
    for i in ['train']:
        new_list=[]
        for j in data_pool['annotations']:
            if j['video_id'] in video_dict[i]:
                for k in data_pool['images']:
                    if k['video_id']==j['video_id'] and j['image_id']==k['id'] :
                        j['frame_index']=k['frame_index']
                        break
                new_list.append(j)
        annotation_dict[i]=new_list

    for i in ['train']:
        overall[i]['annotations']=annotation_dict[i]

    category_mapping = {i['id']: i['name'] for i in data_pool['categories']}
    overall['categories']=category_mapping

    temp =dict()
    for i in ['train']:
        array=[]
        for j in overall[i]['annotations']:
            for k in overall[i]['videos']:
                if j['video_id']==k['id']:
                    res = {**j, **k}
                    break
            array.append(res)
        temp[i]=array
    
    for i in ['train']:
        overall[i]=temp[i]
    
    overall_dict=defaultdict()
    overall_dict['categories']=overall['categories']

    temp=ddddd()
    for info in overall['train']:
        extracted_caption=  [i for i in info['captions'] if i != None]
        for caption in extracted_caption:
            temp_info=info.copy()
            temp_info['captions']=caption
            temp['data'][info['video_name']]["{}-{}".format(info['track_id'],caption)][info['frame_index']].append(temp_info)

    overall_dict['data']=temp['data']
    return overall_dict

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

def extract_frame_image_from_video(video_path:str,frame_ids):
    cap = cv2.VideoCapture(video_path)
    frames = defaultdict()
    H,W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("path: {}, exist path: {}, frame_ids : {}".format(video_path,os.path.exists(video_path),frame_ids))
    for i in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[i] = Image.fromarray(image)
    frames['H'], frames['W'] = H, W
    return frames

def kk():
    return defaultdict(dict)

def kkk():
    return defaultdict(kk)


def multi_dim_dict(n, types):
   
    return defaultdict(ddd)
class BDD_IDUNK(Dataset):
    """
    For the `car` + `color+direction+location` settings
    For the `car` + 'status' settings
    """
    def __init__(self, mode, config, only_car=False):
        super().__init__()
        assert mode in ('train', 'test')
        self.opt = config
        self.mode = mode
        self.only_car = only_car  # 选择类别
        random.seed(config["SEED"])
        self.eliminate_list=['b262f576-b0373824']
        self.overall=standardlize(config["BDD_JSON_PATH"],config["BDD_DATA_ROOT"])
        for k in  self.eliminate_list:
            if k in self.overall['data']:
                del self.overall['data'][k]
        test=random.sample(list(self.overall['data'].keys()), len(self.overall['data'].keys())//5)
        train=[x for x in list(self.overall['data'].keys()) if x not in test]
        self.videos=dict({'test':test,
            'train':train
        })
        self.transform = {idx: get_transform(mode, self.opt, idx) for idx in (0, 1, 2)}
        # self.exp_key = 'expression_new'  # 经处理后的expression标签
        self.frame_data=self._parse_videos()
        self.data = self._parse_data()
        self.data_keys = list(self.data.keys())
        # self.exp2id = {exp: idx for idx, exp in ID2EXP.items()}
    def _parse_videos(self):
        temp=kkk()
        count=0
        for title in self.videos[self.mode]:
            print("Extracted {}/{}".format(count,len(self.videos[self.mode])))
            count=count+1
            key=[]
            for k in  self.overall['data'][title].keys():
                key.extend(list(self.overall['data'][title][k].keys()))

            key=list(dict.fromkeys(key))
            temp[title]=extract_frame_image_from_video(os.path.join(self.opt['BDD_DATA_ROOT'],'train','BDD','{}.mov'.format(title)),key)
        return temp
    def _parse_data(self):
        # labels = json.load(open(self.opt["rf_kitti_json"]))
        data = multi_dim_dict(2, list)
        target_expressions = defaultdict(list)
        for video in self.videos[self.mode]: 
            # load data
            H, W = self.frame_data[video]['H'],self.frame_data[video]['W']

            for key in self.overall['data'][video].keys():
                obj_id = key.split("-")[0]
                num = len(self.overall['data'][video][key].keys())
                if num < self.opt["sample_frame_len"]:
                    continue

                obj_key = f'{video}_{obj_id}'
                pre_frame_id = -1
                curr_data = defaultdict(list)
                for frame_id in self.overall['data'][video][key].keys():
                    # check that the `frame_id` is in order
                    frame_id = int(frame_id)
                    assert frame_id > pre_frame_id
                    pre_frame_id = frame_id
                    for info in self.overall['data'][video][key][frame_id]:
                        # load box
                        exps = [info['captions']]
                        x, y, w, h = info['bbox']
                        # save
                        curr_data['expression'].append(exps)
                        curr_data['target_expression'].append(exps)
                        curr_data['target_labels'].append(1)
                        curr_data['bbox'].append([frame_id, x , y , x + w, y + h])
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
        print("videos: {} __getitem__ {} w {} array: {}".format(video,self.frame_data[video].keys(),sampled_indices,[data['bbox'][idx][0] for idx in sampled_indices]))

        # load images
        images = [
           self.frame_data[video][data['bbox'][idx][0]] for idx in sampled_indices
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
                1
                for i in sampled_target_idx
            ]
            # exp_id = self.exp2id[sampled_target_exp[0]]
        elif self.mode == 'test':
            sampled_target_exp = target_expressions
            sampled_target_label = target_labels
            # exp_id = -1

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
            # expression_id=exp_id,
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

def dummy_transforms():
    return [ T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

def build_bdd_idunk(config: dict, mode: str):
    return BDD_IDUNK(
            config=config,
            mode=mode,
            # transform=dummy_transforms()
            # transform=transforms_for_train(
            #     coco_size=config["COCO_SIZE"],
            #     overflow_bbox=config["OVERFLOW_BBOX"],
            #     reverse_clip=config["REVERSE_CLIP"]
            # )
        )
def get_bdd_dataloader(mode, opt, show=False, **kwargs):
    dataset = build_bdd_idunk(opt, mode)
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
