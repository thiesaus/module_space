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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed


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
        self.cross= nn.CrossEntropyLoss()

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
            "cross_image_text": torch.zeros(()).to(self.device).requires_grad_(),
            "cross_text_image": torch.zeros(()).to(self.device).requires_grad_(),
        }

    def get_sum_loss_dict(self, loss_dict: dict):
        def get_weight(loss_name):
            if "cross_image_text" in loss_name:
                return self.weight["cross_image_text"]
            elif "cross_text_image" in loss_name:
                return self.weight["cross_text_image"]
        
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
        cross_image_text = sum([ self.get_cross_image_text_loss(outputs=out) for out in model_out ])

        # 2. Compute the mae loss.
        cross_text_image = sum([ self.get_cross_text_image_loss(outputs=out) for out in model_out ]) 

        self.loss["cross_image_text"] = self.loss["cross_image_text"]+  cross_image_text
        self.loss["cross_text_image"] =self.loss["cross_image_text"]+  cross_text_image
        # Update logs.
        self.log[f"batch{batch_idx}_cross_image_text"] = cross_image_text.item()
        self.log[f"batch{batch_idx}_cross_text_image"] = cross_text_image.item()

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
       
        loss =  nn.MSELoss()(logits, target)
        return loss

    @staticmethod
    def get_mae_loss(outputs):
        """
        Computer the bounding box loss, l1 and giou.
        """
        logits=outputs
        shape = logits.shape
        target= torch.ones(shape[0],shape[1],dtype=torch.float32).to(outputs.device)
       
        loss =  nn.L1Loss()(logits, target)
        return loss

    # @staticmethod
    def get_cross_image_text_loss(self,outputs):
        """
        Computer the bounding box loss, l1 and giou.
        """
        logits=outputs 
        shape = logits.shape
        target= torch.ones(shape[0],shape[1],dtype=torch.float32).to(outputs.device)
        loss= self.cross(logits, target)
        # loss= self.cross(logits, torch.arange(logits.shape[0],device=logits.device))
        return loss
    
    
    # @staticmethod
    def get_cross_text_image_loss(self,outputs):
        logits=outputs.t() 
        shape = logits.shape
        target= torch.ones(shape[0],shape[1],dtype=torch.float32).to(outputs.device)
        loss= self.cross(logits, target)
        # loss= self.cross(logits, torch.arange(logits.shape[0],device=logits.device))
        return loss


def build_criterion(config: dict):
  
    return ModuleCriterion(
        weight={
            "cross_text_image": 1,
            "cross_image_text": 1,
        },
    )
