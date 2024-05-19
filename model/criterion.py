    # Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import copy
from torch.nn import MSELoss,L1Loss
import torch.nn.functional as F
import torch.distributed

from typing import List, Tuple, Dict

from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy, box_iou_union
from utils.utils import is_distributed, distributed_world_size
from PIL import Image


class ModuleCriterion:
    def __init__(self,  weight: dict):
        """
        Init a criterion function.

        Args:
            weight: include "mse_loss","mae_loss"
        """
        self.device: None | torch.device = None
        self.weight = weight
        self.loss = {}
        self.log = {}
        self.n_logits=[]

    def set_device(self, device: torch.device):
        self.device = device

    def init_module(self, device: torch.device):
        """
        Init this function for a specific clip.
        Args:
            batch: a batch data.
            device:
        Returns:
        """
        self.device = device
        self.n_logits=[]
        self.loss = {
            "mse_loss": torch.zeros(()).to(self.device).requires_grad_(),
            "mae_loss": torch.zeros(()).to(self.device).requires_grad_(),
        }

    def get_sum_loss_dict(self, loss_dict: dict):
        def get_weight(loss_name):
            if "mse_loss" in loss_name:
                return self.weight["mse_loss"]
            elif "mae_loss" in loss_name:
                return self.weight["mae_loss"]
        
        loss = sum([
            get_weight(k) * v for k, v in loss_dict.items()
        ])/2
        return loss.requires_grad_()

    
    def process(self, model_outputs: dict,batch_idx:int):
        """
        Process this criterion for a single frame.

        I know this part is really complex and hard to understand (T.T),
        I will modify these in a possible extension version of this work in the future,
        but it works, doesn't it? :)
        Args:
            model_outputs: outputs from Filter_Module.
        """
        model_out=model_outputs["logits"]
        # 1. Compute the mse loss
        mse_loss = sum([ self.get_mse_loss(outputs=out) for out in model_out ]) / len(model_out)

        # 2. Compute the mae loss.
        mae_loss = sum([ self.get_mse_loss(outputs=out) for out in model_out ]) / len(model_out) 

        self.loss["mse_loss"] = self.loss["mse_loss"] + mse_loss
        self.loss["mae_loss"] = self.loss["mae_loss"] + mae_loss
        # Update logs.
        self.log[f"batch{batch_idx}_mse_loss"] = mse_loss.item()
        self.log[f"batch{batch_idx}_mae_loss"] = mae_loss.item()

        return 
    def get_loss_and_log(self):
        """
        Get the loss and log.
        """
        loss = self.loss
        log = self.log
        return loss, log

    @staticmethod
    def get_mse_loss( outputs):
        """
        Compute the classification loss.
        """
        logits=outputs
        shape = logits.shape
        target= torch.ones(shape[0],shape[1],dtype=torch.float32).to(outputs.device)
       
        loss =  MSELoss()(logits, target)
        return loss

    @staticmethod
    def get_mae_loss(self,outputs):
        """
        Computer the bounding box loss, l1 and giou.
        """
        logits=outputs
        shape = logits.shape
        target= torch.ones(shape[0],shape[1],dtype=torch.float32).to(outputs.device)
       
        loss =  L1Loss()(logits, target)
        return loss

def build_criterion(config: dict):
  
    return ModuleCriterion(
        weight={
            "mse_loss": 1,
            "mae_loss": 1,
        },
    )
