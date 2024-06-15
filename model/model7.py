import math
import numpy as np
from os.path import join

import torch
from torch import nn, randn
import torch.nn.functional as F
from utils.utils import distributed_rank
from einops import rearrange
from transformers import AutoImageProcessor, Swinv2Model, AutoTokenizer,  RobertaModel
import qrcode
from qrcode.image.pure import PyPNGImage
import torchvision.transforms as T
from model.func import MLP, CausalSelfAttention,AddNorm,Weird_Attention


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)

class FFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.mlp(x)
        x = x + self.drop(y)
        x = self.norm(x)
        return x

class TextSelfAttentionBlock(nn.Module):
    def __init__(self, d_model,seq_length, num_heads=4, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.ln_1 = nn.LayerNorm(self.d_model, bias=True)
        self.attn = CausalSelfAttention(d_model,seq_length, num_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(self.d_model, bias=True)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ProcessLayer(nn.Module):
    def __init__(self,d_model,n_head=4,dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.weird_attn = Weird_Attention(d_model,n_head,dropout=dropout)
        
        self.total_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,batch_first=True)
        self.add_norm = AddNorm(d_model,dropout=dropout)
        self.mlp=MLP(d_model,dropout=dropout)

    def forward(self,global_feat,local_feat,text_feat):
        text_feat = rearrange(text_feat,"l b c -> b l c")
        weird_feat = self.weird_attn(global_feat,local_feat,text_feat,True)
        total_feat = self.total_attn(weird_feat,text_feat,text_feat)[0]
        total_feat = self.add_norm(total_feat,weird_feat)
        total_feat = self.mlp(total_feat)
        return total_feat


class Model7(nn.Module):
    def __init__(self):
        super().__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swinv2_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.bert_model=  RobertaModel.from_pretrained("FacebookAI/roberta-base").to(self.device)
        self._freeze_text_encoder()
        #reprocess image

        self.cnn_image = nn.Sequential(*[
            make_layers(1024, 1024, 2, is_downsample=False),
            make_layers(1024, 512, 2, is_downsample=True),
            make_layers(512, 256, 2, is_downsample=True),
        ])
        self.cnn_image_local=nn.Sequential(*[make_layers(1024, 512, 2, is_downsample=False),
                                        make_layers(512, 256, 2, is_downsample=True)])
        self.cnn_image_global=nn.Sequential(*[make_layers(1024, 512, 2, is_downsample=False),
                                        make_layers(512, 256, 2, is_downsample=True)])
     
          #reprocess text
        self.text_linear_phase1=nn.Sequential(*[
            nn.Linear(768, 384).to(self.device),
            nn.GELU(),
            nn.Linear(384, 192).to(self.device),
            nn.GELU(),
            nn.Linear(192, 96).to(self.device),
            nn.GELU(),
            nn.Linear(96, 64).to(self.device)
            ])

        self.text_linear_phase2=nn.Sequential(*[
            nn.Linear(4096, 2048).to(self.device),
            nn.GELU(),
            nn.Linear(2048, 1024).to(self.device),
            nn.GELU(),
            nn.Linear(1024, 512).to(self.device),
            nn.GELU(),
            nn.Linear(512, 256).to(self.device),
        ])

        
        self.feature_dim=256

        self.img_dim = 256
        self.text_dim = 768
        self.img_fc = self.get_img_fc(use_ln=False)
        self.text_fc = self.get_text_fc(use_ln=False)

        local_reso = 8 * 8
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso))
        global_reso = 8 * 8
        global_scale = global_reso ** -0.5
        self.pos_emb_global = nn.Parameter(global_scale * randn(global_reso))
        qr_reso = 8 * 8
        qr_scale = qr_reso ** -0.5
        self.pos_emb_qr = nn.Parameter(qr_scale * randn(qr_reso))
       
        self.fusion_fc = nn.Linear(self.text_dim, self.img_dim)

        self.qr_transform= T.Compose([
            T.ToTensor(),
            T.Resize((672,672)),
        ])

        self.local_attn =nn.MultiheadAttention(self.feature_dim, 4, dropout=0.)
        self.local_add_norm = AddNorm(self.feature_dim, dropout=0.)

        self.global_attn =nn.MultiheadAttention(self.feature_dim, 4, dropout=0.)
        self.global_add_norm = AddNorm(self.feature_dim, dropout=0.)

        self.textual_attn = TextSelfAttentionBlock(self.feature_dim, 64, num_heads=4, dropout=0.)

        self.qr_attn = nn.MultiheadAttention(self.feature_dim, 4, dropout=0.)
        self.qr_add_norm = AddNorm(self.feature_dim, dropout=0.)
    
        self.process_layer = ProcessLayer(self.feature_dim,4,dropout=0.)

    def _freeze_text_encoder(self):
        """
        These parameters are not frozen:
        - list(self.clip.token_embedding.parameters())
        - [self.clip.positional_embedding]
        """
        for p in list(self.bert_model.parameters()) + \
                list(self.swinv2_model.parameters()):
            p.requires_grad = False
        self.bert_model.eval()
        self.swinv2_model.eval()

    def make_qr_image(self,texts):
        imgs=[]
        for text in texts:
            img = qrcode.make(text,box_size=10)
            imgs_tensor =1- self.qr_transform(img.get_image())
            imgs.append(imgs_tensor)
        
        return torch.stack(imgs).repeat(1,3,1,1)
    
    def qr_code_encoder(self,qr_imgs):
        qr_imgs = rearrange(qr_imgs, 'b c h w -> b c h w')
        # local_img=(local_img-  local_img.min())/ (local_img.max() - local_img.min())
        qr_feats =  self.process_image(qr_imgs);  # [bt,c,7,7]
        qr_feat_hidden = rearrange(qr_feats,"b hw c -> b c hw")
        qr_feat_hidden = qr_feat_hidden + self.pos_emb_qr
        qr_feat_hidden = rearrange(qr_feat_hidden,"b c (h w) -> b c h w",h=8)
        qr_feat_hidden = self.cnn_image(qr_feat_hidden)
        qr_feat_hidden = rearrange(qr_feat_hidden,"b c h w -> b (c h w)")
        # qr_imgs = qr_imgs.unsqueeze(1).repeat(1,n,1,1)
        return qr_feat_hidden
    
    def enhance_self_feature(self,local_feat,global_feat,text_feat):
        # local feature
        y1 = self.local_attn(local_feat, local_feat, local_feat)[0]
        y1 = self.local_add_norm(y1, local_feat)
        # global feature
        y2 = self.global_attn(global_feat, global_feat, global_feat)[0]
        y2 = self.global_add_norm(y2, global_feat)
        # textual feature
        y3 = self.textual_attn(text_feat)
        # qr feature
      

        return y1,y2,y3

    def forward(self, x, epoch=1e5):
        output = dict()
        imgs= x['local_images']
        texts = x['sentences']
        labels = None
        if self.training:
            labels = x['labels']
        b,n = imgs.size()[:2]
        # qr_imgs = self.make_qr_image(texts)
        # qr_feats = self.qr_code_encoder(qr_imgs)
        # qr_imgs = rearrange(qr_imgs,"b n c h w -> (b n) c h w")
        textual_hidden, text_feat = self.textual_encoding(texts)

        assert len(text_feat.size()) == 3
        # get textual embeddings
        text_feat = text_feat.unsqueeze(1)  # [b,l,c]->[b,1,l,c]
        text_feat = text_feat.repeat([1, n, 1, 1])
        text_feat = rearrange(text_feat, 'b t l c -> (b t) l c')
        text_feat = self.fusion_fc(text_feat)
        text_feat = rearrange(text_feat, 'bt l c -> l bt c')

        local_feat,global_feat = self.visual_extract(x['local_images'],x['global_image'])

        local_feat,global_feat,text_feat = self.enhance_self_feature(local_feat,global_feat,text_feat)
      

        visual_feat = self.process_layer(global_feat,local_feat,text_feat) * local_feat
        visual_feat = rearrange(visual_feat, 'b l c -> b c l')
        visual_feat = self.st_pooling(visual_feat, b)

        # visual_feat = rearrange(visual_feat,'(b t) c -> t b c',b=b)
        k1 = rearrange(visual_feat,"(b n) c -> n b c",b=b)
        k2 = rearrange(textual_hidden,"(b n) c -> n b c",b=b)
        scores = torch.mean(F.cosine_similarity(k1, k2, dim=-1),0)
        # temp=torch.zeros(scores.shape[1],device=self.device)
        # for i in range(scores.shape[0]):
        #     temp= temp+scores[i]
        # scores=temp/scores.shape[0]
            

        output['scores'] = scores
        output['vis_feat'] = visual_feat
        # output['qr_feat'] = qr_feats
        output['loss']=torch.tensor(0.0,device=self.device)
        return output

    def st_pooling(self, feat, bs):
        # spatial pooling
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [bt,c,l]->[bt,c]
        # feat = rearrange(feat, 'b c t -> (b t) c')
        # temporal pooling
        feat = rearrange(feat, '(b t) c -> b c t', b=bs)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)  # [b,c]
        # projection
        feat = self.img_fc(feat)
        return feat

    def visual_extract(self,local_img,global_img):
        b, t = global_img.size()[:2]
        local_img = rearrange(local_img, 'b t c h w -> (b t) c h w')
        local_feat =  self.process_image(local_img);  # [bt,c,7,7]
      
        global_img = rearrange(global_img, 'B T C H W -> (B T) C H W')
        global_feat =  self.process_image(global_img); 
      
        # rearrange
        local_feat = rearrange(local_feat, 'bt hw c -> bt c hw')
        global_feat = rearrange(global_feat, 'bt HW c -> bt c HW')
        local_feat = local_feat + self.pos_emb_local
        global_feat = global_feat + self.pos_emb_global

        local_feat = rearrange(local_feat, 'bt c (h w) -> bt c h w',h=8)
        local_feat = self.cnn_image_local(local_feat)

        global_feat = rearrange(global_feat, 'BT C (H W) -> BT C H W',H=8)
        global_feat = self.cnn_image_global(global_feat)

        local_feat = rearrange(local_feat, 'bt c h w -> bt (h w) c')
        global_feat = rearrange(global_feat, 'bt c H W -> bt (H W) c')
        return local_feat,global_feat

      
    def textual_encoding(self, texts):
        text=self.text_encoder(texts)
        text_hidden = self.text_linear_phase1(text)
        text_hidden = rearrange(text_hidden,"b w c -> b (w c)")
        text_hidden = self.text_linear_phase2(text_hidden)
        if self.training:
            return text_hidden,text
        else:
            return text_hidden, F.normalize(text, p=2, dim=-1)

    def get_img_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.img_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim, eps=1e-12),
            )
        else:
            return nn.Linear(self.img_dim, self.feature_dim)

    def get_text_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim),
                nn.ReLU(),
                nn.Linear(self.text_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim, eps=1e-12),
            )
        else:
            return nn.Sequential(
                nn.Linear(self.text_dim, self.text_dim),
                nn.ReLU(),
                nn.Linear(self.text_dim, self.feature_dim),
            )

    def text_encoder(self, text):  # [1,3,768]
        inputs = self.tokenizer.batch_encode_plus(text,max_length=64,padding="max_length",  return_special_tokens_mask=True, return_tensors="pt",  truncation=True).to(self.device)
        tokenizer_input = {"input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"]}

        outputs = self.bert_model(**tokenizer_input)
        return outputs.last_hidden_state

    
    def process_image(self,image):

        temp=self.image_encoder(image)
        padded_temp = F.pad(temp, (0, 256, 0, 0, 0, 0), mode='constant', value=0)
        return padded_temp
    
    def image_encoder(self, image): # [1,49,768]
        inputs = self.image_processor(image, return_tensors="pt",do_rescale=False).to(self.device)    
        outputs = self.swinv2_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states 

def build_model7(config: dict):

    model = Model7()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model