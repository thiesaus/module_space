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


class Super_Weird_Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1,batch_first=True):
        super().__init__()
        assert d_model % n_heads == 0, "The hidden size is not a multiple of the number of attention heads"

        self.d_model = d_model
        self.n_heads = n_heads
        
        self.attention_head_size = int(d_model / n_heads)
        self.all_head_size = n_heads * self.attention_head_size

        # global_local block (ltb)
        self.ltb_q = nn.Linear(d_model, self.all_head_size)
        self.ltb_k = nn.Linear(d_model, self.all_head_size)
        self.ltb_v = nn.Linear(d_model, self.all_head_size)

        # prompt_local block (tlb)
        self.tlb_q = nn.Linear(d_model, self.all_head_size)
        self.tlb_k = nn.Linear(d_model, self.all_head_size)
        self.tlb_v = nn.Linear(d_model, self.all_head_size)

        self.ltb_dense = nn.Linear(d_model, d_model)
        self.tlb_dense = nn.Linear(d_model, d_model)
        self.ltb_dropout = nn.Dropout(p=dropout)
        self.tlb_dropout = nn.Dropout(p=dropout)

        self.batch_first=batch_first
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,pair):
        '''
        local_feat: [Seq_length x Batch_size x Hidden_size]
        text_feat: [Seq_length x Batch_size x Hidden_size]
        '''
        local_feat, text_feat = pair
        if self.batch_first == False:
            local_feat = local_feat.permute(1, 0, 2) # [Batch_size x Seq_length x Hidden_size]
            text_feat = text_feat.permute(1, 0, 2)

        # (tlb)
        tlb_mixed_q_layer = self.ltb_q(local_feat)  # [Batch_size x Seq_length x Hidden_size]
        tlb_mixed_k_layer = self.tlb_k(text_feat)  # [Batch_size x Seq_length x Hidden_size]
        tlb_mixed_v_layer = self.tlb_v(text_feat)  # [Batch_size x Seq_length x Hidden_size]

        # (ltb)
        ltb_mixed_q_layer = self.ltb_q(text_feat)
        ltb_mixed_k_layer = self.ltb_k(local_feat)
        ltb_mixed_v_layer = self.ltb_v(local_feat)


        # (tlb)
        tlb_q_layer = self.transpose_for_scores(
            tlb_mixed_q_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        tlb_k_layer = self.transpose_for_scores(tlb_mixed_k_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        tlb_v_layer = self.transpose_for_scores(
            tlb_mixed_v_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        # (ltb)
        ltb_q_layer = self.transpose_for_scores(ltb_mixed_q_layer)
        ltb_k_layer = self.transpose_for_scores(ltb_mixed_k_layer)
        ltb_v_layer = self.transpose_for_scores(ltb_mixed_v_layer)

        # (tlb)
        tlb_attention_scores = torch.matmul(tlb_q_layer, tlb_k_layer.transpose(-1,-2)) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        tlb_attention_scores = tlb_attention_scores / math.sqrt(self.attention_head_size) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        tlb_attention_probs = nn.Softmax(dim=-1)(tlb_attention_scores) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        tlb_attention_probs = self.tlb_dropout(tlb_attention_probs) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        local_q_bridge_value=tlb_attention_probs

        # (ltb)
        ltb_attention_scores = torch.matmul(ltb_q_layer, ltb_k_layer.transpose(-1,-2))
        ltb_attention_scores = ltb_attention_scores / math.sqrt(self.attention_head_size)
        ltb_attention_probs = nn.Softmax(dim=-1)(ltb_attention_scores)
        ltb_attention_probs = self.ltb_dropout(ltb_attention_probs)
        text_q_bridge_value=ltb_attention_probs

        ltb_context_enhanced=torch.matmul(local_q_bridge_value,ltb_attention_probs)
        tlb_context_enhanced=torch.matmul(text_q_bridge_value,tlb_attention_probs)

        # add value
        ltb_context_layer = torch.matmul(ltb_context_enhanced, ltb_v_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        tlb_context_layer = torch.matmul(tlb_context_enhanced, tlb_v_layer)


        # glb context layer
        tlb_context_layer = tlb_context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        tlb_new_context_layer_shape = tlb_context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        tlb_context_layer = tlb_context_layer.view(*tlb_new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]


        # plb context layer
        ltb_context_layer = ltb_context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        ltb_new_context_layer_shape = ltb_context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        ltb_context_layer = ltb_context_layer.view(*ltb_new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        new_local=self.ltb_dense(ltb_context_layer)
        new_text=self.tlb_dense(tlb_context_layer)
        return (new_local,new_text)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.):
        super().__init__()
        self.attn = Super_Weird_Attention(d_model, n_heads, dropout=dropout)
        self.local_add_norm = AddNorm(d_model, dropout=dropout)
        self.local_ffn = FFN(d_model, dropout=dropout)
        self.local_add_norm_2 = AddNorm(d_model, dropout=dropout)

        self.text_add_norm = AddNorm(d_model, dropout=dropout)
        self.text_ffn = FFN(d_model, dropout=dropout)
        self.text_add_norm_2 = AddNorm(d_model, dropout=dropout)

    def forward(self, pair):
        local_feat, text_feat = pair
        new_local_feat, new_text_feat = self.attn((local_feat, text_feat))

        new_local_feat = self.local_add_norm(new_local_feat, local_feat)
        new_local_feat = self.local_ffn(new_local_feat)
        new_local_feat = self.local_add_norm_2(new_local_feat, new_local_feat)

        new_text_feat = self.text_add_norm(new_text_feat, text_feat)
        new_text_feat = self.text_ffn(new_text_feat)
        new_text_feat = self.text_add_norm_2(new_text_feat, new_text_feat)
        
        return (new_local_feat, new_text_feat)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_blocks, n_heads=4, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderBlock(d_model, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, pair):
        return self.layers(pair)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)
        self.add_norm = AddNorm(d_model, dropout=dropout)
        self.local_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)
        self.local_add_norm = AddNorm(d_model, dropout=dropout)
        self.text_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)
        self.text_add_norm = AddNorm(d_model, dropout=dropout)
        self.ffn = FFN(d_model, dropout=dropout)
        self.ffn_add_norm = AddNorm(d_model, dropout=dropout)
    def forward(self,pair):
        global_feat, local_feat, text_feat = pair
        new_global_feat, _ = self.self_attn(global_feat, global_feat, global_feat)
        new_global_feat = self.add_norm(new_global_feat, global_feat)

        new_global_local_feat, _ = self.local_cross_attn(new_global_feat, local_feat, local_feat)
        new_global_local_feat = self.local_add_norm(new_global_local_feat, new_global_feat)

        new_global_text_feat, _ = self.text_cross_attn(new_global_local_feat, text_feat, text_feat)
        new_global_text_feat = self.text_add_norm(new_global_text_feat, new_global_local_feat)

        new_global_text_feat = self.ffn(new_global_text_feat)
        new_global_text_feat = self.ffn_add_norm(new_global_text_feat, new_global_text_feat)
        return (new_global_text_feat,local_feat,text_feat)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_blocks, n_heads=4, dropout=0.):
        super().__init__()
        self.layers = nn.Sequential(*[DecoderBlock(d_model, n_heads, dropout) for _ in range(n_blocks)])

    def forward(self, pair):
        return self.layers(pair)


class Weird_Fusion_Module(nn.Module):
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

        self.local_attn =nn.MultiheadAttention(self.feature_dim, 4, dropout=0.,batch_first=True)
        self.local_add_norm = AddNorm(self.feature_dim, dropout=0.)

        self.global_attn =nn.MultiheadAttention(self.feature_dim, 4, dropout=0.,batch_first=True)
        self.global_add_norm = AddNorm(self.feature_dim, dropout=0.)

        self.textual_attn = TextSelfAttentionBlock(self.feature_dim, 64, num_heads=4, dropout=0.)
    
        self.encoder_layer = EncoderLayer(self.feature_dim, 4, n_heads=4, dropout=0.)
        self.decoder_layer = DecoderLayer(self.feature_dim, 1, n_heads=4, dropout=0.)

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
        # text_feat = rearrange(text_feat, 'bt l c -> l bt c')

        local_feat,global_feat = self.visual_extract(x['local_images'],x['global_image'])

        local_feat,global_feat,text_feat = self.enhance_self_feature(local_feat,global_feat,text_feat)
      

        local_feat,text_feat = self.encoder_layer((local_feat,text_feat))
        decoded_feat,local_feat,text_feat = self.decoder_layer((global_feat,local_feat,text_feat))

        visual_feat= decoded_feat * local_feat
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

def build_weird_fusion_module(config: dict):

    model = Weird_Fusion_Module()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model