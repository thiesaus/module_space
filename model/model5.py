import math
import numpy as np
from os.path import join

import torch
from torch import nn, randn
import torch.nn.functional as F
from utils.utils import distributed_rank
from einops import rearrange
from transformers import AutoImageProcessor, Swinv2Model, AutoTokenizer,  RobertaModel

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False).requires_grad_()
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False).requires_grad_()
        self.bn1 = nn.BatchNorm2d(c_out).requires_grad_()
        self.relu = nn.ReLU(True).requires_grad_()
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False).requires_grad_()
        self.bn2 = nn.BatchNorm2d(c_out).requires_grad_()
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False).requires_grad_(),
                nn.BatchNorm2d(c_out).requires_grad_()
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False).requires_grad_(),
                nn.BatchNorm2d(c_out).requires_grad_()
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


class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swinv2_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.bert_model=  RobertaModel.from_pretrained("FacebookAI/roberta-base").to(self.device)
         #reprocess image
        self.reprocess_image=make_layers(1024, 512, 2, is_downsample=False)
        self.reprocess_image1=make_layers(512, 512, 2, is_downsample=True)
        self.reprocess_image2=make_layers(512, 256, 2, is_downsample=True)
        self.reprocess_image3=make_layers(256 , 256 , 2, is_downsample=True)

        #reprocess text
        self.text_linear = nn.Linear(768, 384).to(self.device).requires_grad_()
        self.text_linear1 = nn.Linear(384, 192).to(self.device).requires_grad_()
        self.text_linear2 = nn.Linear(192, 96).to(self.device).requires_grad_()
        self.text_linear3 = nn.Linear(96, 64).to(self.device).requires_grad_()
        self.text_linear4 = nn.Linear(4096, 2048).to(self.device).requires_grad_()
        self.text_linear5 = nn.Linear(2048, 1024).to(self.device).requires_grad_()
        # self.text_linear6 = nn.Linear(1024, 512).to(self.device).requires_grad_()
        # self.text_linear7 = nn.Linear(512, 256).to(self.device).requires_grad_()
        self.feature_dim=1024

        self.img_dim = 1024
        self.text_dim = 768
        self.img_fc = self.get_img_fc(use_ln=False)
        self.text_fc = self.get_text_fc(use_ln=False)
        # self._freeze_text_encoder()

        self.fusion_local_global = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0.,
        )

        local_reso = 8 * 8
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso))
        global_reso = 8 * 8
        global_scale = global_reso ** -0.5
        self.pos_emb_global = nn.Parameter(global_scale * randn(global_reso))

        # if self.opt.kum_mode == 'cascade attention':
        self.fusion_visual_textual = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0,
        )
        self.fusion_fc = nn.Linear(self.text_dim, self.img_dim)
        self.fusion_ffn = FFN(self.img_dim, 0.1)
        #     self.fusion_drop = nn.Dropout(p=0.1)
        # elif self.opt.kum_mode in ('cross correlation', 'text-first modulation'):
        #     self.fusion_conv1 = nn.Sequential(
        #         nn.Conv1d(self.text_dim, self.img_dim, kernel_size=1, bias=False),
        #         nn.BatchNorm1d(self.img_dim),
        #     )
        #     self.fusion_conv2 = nn.Sequential(
        #         nn.Conv1d(self.img_dim, self.img_dim, kernel_size=1, bias=False),
        #         nn.BatchNorm1d(self.img_dim),
        #     )
        #     self.fusion_drop = nn.Dropout(p=0.1)
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


    def _init_weights_function(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        else:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, epoch=1e5):
        output = dict()
        textual_hidden, textual_feat = self.textual_encoding(x['sentences'])
        # if self.opt.kum_mode and (epoch >= self.opt.tg_epoch):
        #     if self.opt.kum_mode == 'cascade attention':
        #         visual_feat = self.visual_local_global(
        #             x['local_img'], x['global_img'], textual_hidden, self.opt.kum_mode
        #         )
        #     elif self.opt.kum_mode in ['cross correlation', 'text-first modulation']:
        #         visual_feat = self.visual_local_global(
        #             x['local_img'], x['global_img'], textual_feat, self.opt.kum_mode
        #         )
        # else:
        #     visual_feat = self.visual_local_global(x['local_img'], x['global_img'])
        visual_feat = self.visual_local_global(
                    x['local_images'], x['global_image'], textual_feat
                )
        b= x['local_images'].shape[0]
        visual_feat = rearrange(visual_feat,'(b t) c -> t b c',b=b)
        logits = F.cosine_similarity(visual_feat, textual_hidden,dim=-1)
        temp=torch.zeros(logits.shape[1],device=self.device,requires_grad=True)
        for i in range(logits.shape[0]):
            temp= temp+logits[i]
        logits=temp/logits.shape[0]
            

        output['logits'] = torch.tensor(logits, device=self.device, requires_grad=True)
        output['vis_feat'] = visual_feat
        output['text_feat'] = textual_feat
        return output

    def st_pooling(self, feat, bs):
        # spatial pooling
        feat = F.adaptive_avg_pool1d(feat, 1)  # [bt,c,l]->[bt,c]
        # feat = rearrange(feat, 'b c t -> (b t) c')
        # temporal pooling
        # feat = rearrange(feat, '(b t) c -> b c t', b=bs)
        feat = F.adaptive_avg_pool1d(feat, 1)  # [b,c]
        feat = rearrange(feat, 'b c t -> (b t) c')

        # projection
        feat = self.img_fc(feat)
        return feat

    def cross_modal_fusion(self, vis_feat, text_feat, b,t):
        # if mode == 'cascade attention':
        assert len(text_feat.size()) == 3
        # get textual embeddings
        text_feat = text_feat.unsqueeze(1)  # [b,l,c]->[b,1,l,c]
        text_feat = text_feat.repeat([1, t, 1, 1])
        text_feat = rearrange(text_feat, 'b t l c -> (b t) l c')
        text_feat = self.fusion_fc(text_feat)
        text_feat = rearrange(text_feat, 'bt l c -> l bt c')
        # fusion
        fused_feat = self.fusion_visual_textual(
            query=vis_feat,
            key=text_feat,
            value=text_feat,
        )[0]
        vis_feat = vis_feat * fused_feat
        vis_feat = rearrange(vis_feat, 'l bt c -> bt c l')
        return vis_feat
        # elif mode == 'cross correlation':
        #     assert len(text_feat.size()) == 2
        #     # get textual embeddings
        #     text_feat = text_feat.unsqueeze(1)  # [b,c]->[b,1,c]
        #     text_feat = text_feat.repeat([1, t, 1])  # [b,t,c]
        #     text_feat = rearrange(text_feat, 'b t c -> (b t) c 1')  # [bt,c,1]
        #     text_feat = self.fusion_conv1(text_feat)  # [bt,c,1]
        #     # fusion
        #     vis_feat = rearrange(vis_feat, 'HW bt c -> bt c HW')  # [bt,c,l]
        #     fused_feat = xcorr_depthwise(vis_feat, kernel=text_feat)  # [bt,c,l]
        #     vis_feat = vis_feat + self.fusion_drop(fused_feat)
        #     vis_feat = self.fusion_conv2(vis_feat)
        #     return vis_feat
        # elif mode == 'text-first modulation':
        #     assert len(text_feat.size()) == 2
        #     L, _, _ = vis_feat.size()
        #     # get textual embeddings
        #     text_feat = text_feat.unsqueeze(1)  # [b,c]->[b,1,c]
        #     text_feat = text_feat.repeat([1, t, 1])  # [b,t,c]
        #     text_feat = rearrange(text_feat, 'b t c -> (b t) c 1')  # [bt,c,1]
        #     text_feat = self.fusion_conv1(text_feat)  # [bt,c,1]
        #     text_feat = text_feat.repeat([1, 1, L])  # [bt,c,HW]
        #     # fusion
        #     vis_feat = rearrange(vis_feat, 'HW bt c -> bt c HW')
        #     out_feat = vis_feat * self.fusion_drop(text_feat)
        #     out_feat = rearrange(out_feat, 'bt c HW -> HW bt c')
        #     return out_feat

    def visual_local_global(self, local_img, global_img, text_feat=None):
        # b, t = global_img.size()[:2]
        # spatial encoding
        # _global_feat=rearrange(_global_feat,"b (h w) c -> b c h w",h=8)
        b, t = global_img.size()[:2]
        local_img = rearrange(local_img, 'b t c h w -> (b t) c h w')
        local_img=(local_img-  local_img.min())/ (local_img.max() - local_img.min())
        local_feat =  self.process_image(local_img);  # [bt,c,7,7]
        # local_feat=rearrange(local_feat,"b (h w) c -> b c h w",h=8)
        # bt, c, h, w = local_feat.size()
        global_img = rearrange(global_img, 'B T C H W -> (B T) C H W')
        global_img=(global_img-  global_img.min())/ (global_img.max() - global_img.min())

        global_feat =  self.process_image(global_img); 
        # global_feat = torch.stack([global_feat for _ in range(b)], dim=1).squeeze(0)
        # global_feat=rearrange(global_feat,"b (h w) c -> b c h w",h=8)

        # global_img = rearrange(global_img, 'B T C H W -> (B T) C H W')
        # global_feat = self.clip.visual(global_img, with_pooling=False)  # [bt,c,7,7]
        # bt, c, H, W = global_feat.size()
        # rearrange
        local_feat = rearrange(local_feat, 'bt hw c -> bt c hw')
        global_feat = rearrange(global_feat, 'bt HW c -> bt c HW')
        local_feat = local_feat + self.pos_emb_local
        global_feat = global_feat + self.pos_emb_global
        local_feat = rearrange(local_feat, 'bt c hw -> hw bt c')
        global_feat = rearrange(global_feat, 'bt c HW -> HW bt c')
        # text-guided
        # if kum_mode == 'text-first modulation':
        #     local_feat_2 = self.cross_modal_fusion(
        #         local_feat, text_feat, b, t, kum_mode
        #     )
        #     global_feat_2 = self.cross_modal_fusion(
        #         global_feat, text_feat, b, t, kum_mode
        #     )
        #     fusion_feat = self.fusion_local_global(
        #         query=local_feat_2,
        #         key=global_feat_2,
        #         value=global_feat,
        #     )[0]
        # else:
            # cross-attention
        fusion_feat = self.fusion_local_global(
            query=local_feat,
            key=global_feat,
            value=global_feat,
        )[0]
        fusion_feat = fusion_feat + local_feat  # [HW,bt,c]
        # text-guided
        # if kum_mode in ('cascade attention', 'cross correlation'):
        fusion_feat= self.cross_modal_fusion(
            fusion_feat, text_feat, b,t
        )
        # else:
        #     fusion_feat = rearrange(fusion_feat, 'HW bt c -> bt c HW')
        fusion_feat = self.st_pooling(fusion_feat, bs=b)
        if self.training:
            return fusion_feat
        else:
            fusion_feat = F.normalize(fusion_feat, p=2, dim=-1)
            return fusion_feat

    def textual_encoding(self, texts):
        # x_hidden, x = self.clip.encode_text_2(tokens, self.opt.truncation)
        # x = self.text_fc(x)
        text=self.text_encoder(texts)
        text_hidden = self.text_linear(text)
        text_hidden = self.text_linear1(text_hidden)
        text_hidden = self.text_linear2(text_hidden)
        text_hidden = self.text_linear3(text_hidden)
        text_hidden = rearrange(text_hidden,"b w c -> b (w c)")
        text_hidden = self.text_linear4(text_hidden)
        text_hidden = self.text_linear5(text_hidden)
        # text = self.text_linear6(text)
        # text = self.text_linear7(text)
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
    
    def cnn_image(self, local_image):
        local_image = self.reprocess_image(local_image)
        local_image = self.reprocess_image1(local_image)
        local_image = self.reprocess_image2(local_image)
        local_image = self.reprocess_image3(local_image)
        return local_image
    
def build_model5(config: dict):

    model = Model5()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model