import math
import numpy as np
from os.path import join

import torch
from torch import nn, randn
import torch.nn.functional as F
from utils.utils import distributed_rank
from einops import rearrange
from transformers import AutoImageProcessor, Swinv2Model, AutoTokenizer,  RobertaModel
import torchvision.transforms as T
# from model.func import MLP, CausalSelfAttention,AddNorm,Weird_Attention
from model.transformers.vanilla import TransformerEncoderLayer
from model.transformers.fuses_modules import BiAttentionBlock
from model.transformers.utils import _get_clones, get_sine_pos_embed,_get_activation_fn, gen_sineembed_for_position,MLP

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


class Img_Encoder_Layer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_heads=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)
      
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation, d_model=dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else (tensor.permute(0,2,1) + pos).permute(0,2,1)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, feat,pos=None
    ):
        # self attention
        # import ipdb; ipdb.set_trace()
        query= key= value=self.with_pos_embed(feat, pos)

        src2 = self.self_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            key_padding_mask=None,
        )[0]
           
        src = feat + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4,num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.text_enhance_layer = TransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
   
        self.enhance_global= Img_Encoder_Layer(
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.feature_fusion_layer = BiAttentionBlock(
            v_dim=d_model,
            l_dim=d_model,
            embed_dim=dim_feedforward ,
            num_heads=n_heads ,
            dropout=dropout,
            drop_path=0.1,
        )
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        self.layers = _get_clones(self.enhance_global, num_layers, layer_share=None)

        self.text_layers = _get_clones(
            self.text_enhance_layer, num_layers, layer_share=None
        )
        self.fusion_layers = _get_clones(
            self.feature_fusion_layer, num_layers, layer_share=None
        )
        local_reso = 4 * 4
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso))
 
       
    def forward(self,global_feat,text_feat):
        bs, n_text, text_dim = text_feat.shape
        pos_text = (
                    torch.arange(n_text, device=text_feat.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .repeat(bs, 1, 1)
                )
        pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
        memory_text = text_feat
        output = global_feat
        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            
            output, memory_text = self.fusion_layers[layer_id](
                v=output,
                l=memory_text,
                attention_mask_v=None,
                attention_mask_l=None,
            )

            memory_text = self.text_layers[layer_id](
                src=memory_text,
                src_mask=None,  # note we use ~ for mask here
                src_key_padding_mask=None,
                pos=(pos_text if pos_text is not None else None),
            )

            # main process
         
            output = layer(
                feat=output,
                pos=self.pos_emb_local,
            )

        return output, memory_text
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model, n_heads=4, dim_feedforward=1024, dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation, d_model=dim_feedforward, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    
    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor+ pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        feature,
        query_pos,
        global_feature,
        text_feature,
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
      

        # self attention
            # import ipdb; ipdb.set_trace()
        q = k = self.with_pos_embed(feature, query_pos)
        tgt2 = self.self_attn(q, k, feature)[0]
        tgt = feature + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.ca_text(
            self.with_pos_embed(tgt, query_pos),
            text_feature.transpose(0, 1),
            text_feature.transpose(0, 1),
        )[0]
        tgt = tgt + self.catext_dropout(tgt2)
        tgt = self.catext_norm(tgt)

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
            key=global_feature,
            value=global_feature,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DecoderLayer(nn.Module):
    def __init__(self, d_model,num_layer, n_heads=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.layers = _get_clones(
            DecoderBlock(d_model, n_heads, dim_feedforward, dropout),
            num_layer,
            layer_share=None,
        )
        self.ref_point_head = MLP(d_model, d_model, d_model, 2)
        self.norm=nn.LayerNorm(d_model)
    def forward(self, feature, global_feature, text_feature):
        feature = rearrange(feature, 'b l c -> l b c')
        query_pos = gen_sineembed_for_position(feature[:,:,0:2])
        raw_query_pos = self.ref_point_head(query_pos) 

        for layer in self.layers:
            feature = layer(feature,raw_query_pos, global_feature, text_feature)
        return  self.norm(feature)

class GDino_Clone(nn.Module):
    def __init__(self):
        super().__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim=256
        self.img_dim = 256
        self.text_dim = 768
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swinv2_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.bert_model=  RobertaModel.from_pretrained("FacebookAI/roberta-base").to(self.device)
        self._freeze_text_encoder()

        # text process
        self.feat_map = nn.Linear(self.text_dim, self.feature_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
     
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

        local_reso = 8 * 8
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso))
 
        self.encoder_layer = EncoderLayer(d_model=self.feature_dim, n_heads=4,num_layers=6, dim_feedforward=1024, dropout=0.1)
        self.decoder_layer=DecoderLayer(d_model=self.feature_dim,num_layer=6, n_heads=4, dim_feedforward=1024, dropout=0.1)

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

    def st_pooling(self, feat, bs):
        # spatial pooling
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [bt,c,l]->[bt,c]
        # temporal pooling
        feat = rearrange(feat, '(b t) c -> b c t', b=bs)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [b,c]

        return feat


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
        text_features=self.text_encoder(texts)
        text_feat = self.feat_map(text_features)

        assert len(text_feat.size()) == 3
        # get textual embeddings
        text_feat = text_feat.unsqueeze(1)  # [b,l,c]->[b,1,l,c]
        text_feat = text_feat.repeat([1, n, 1, 1])
        text_feat = rearrange(text_feat, 'b t l c -> (b t) l c')

        local_feat,global_feat = self.visual_extract(x['local_images'],x['global_image'])

        new_global_feat,new_text_feat = self.encoder_layer(global_feat,text_feat)

        # get the final visual features
        fused_feat = self.decoder_layer(local_feat, new_global_feat, new_text_feat)


        visual_feat =   fused_feat  
        visual_feat = rearrange(visual_feat, 'l b c -> b c l')
        visual_feat = self.st_pooling(visual_feat, b)

        compare_text_feat = new_text_feat
        compare_text_feat = rearrange(compare_text_feat, 'b l c -> b c l')
        compare_text_feat = self.st_pooling(compare_text_feat, b)

        # visual_feat = rearrange(visual_feat,'(b t) c -> t b c',b=b)
        k1 = rearrange(visual_feat,"(b n) c -> n b c",b=b)
        k2 = rearrange(compare_text_feat,"(b n) c -> n b c",b=b)
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

    def visual_extract(self,local_img,global_img):
        b, t = global_img.size()[:2]
        local_img = rearrange(local_img, 'b t c h w -> (b t) c h w')
        local_feat =  self.process_image(local_img);  # [bt,c,7,7]
      
        global_img = rearrange(global_img, 'B T C H W -> (B T) C H W')
        global_feat =  self.process_image(global_img); 
      
        # rearrange
        local_feat = rearrange(local_feat, 'bt hw c -> bt c hw')
        global_feat = rearrange(global_feat, 'bt HW c -> bt c HW')
        # local_feat = local_feat + self.pos_emb_local
        # global_feat = global_feat + self.pos_emb_global

        local_feat = rearrange(local_feat, 'bt c (h w) -> bt c h w',h=8)
        local_feat = self.cnn_image_local(local_feat)

        global_feat = rearrange(global_feat, 'BT C (H W) -> BT C H W',H=8)
        global_feat = self.cnn_image_global(global_feat)

        local_feat = rearrange(local_feat, 'bt c h w -> bt (h w) c')
        global_feat = rearrange(global_feat, 'bt c H W -> bt (H W) c')
        return local_feat,global_feat

      

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

def build_gdino_clone(config: dict):

    model = GDino_Clone()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model