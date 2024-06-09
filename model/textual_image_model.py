import torch.nn as nn
import torch
from transformers import AutoImageProcessor, Swinv2Model, RobertaTokenizerFast, RobertaModel,  DeformableDetrModel
from utils.utils import distributed_rank
from einops import rearrange,repeat
from torch.nn import functional as F
from model.position_embedding import build


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.01):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.layer_norm(self.dropout(Y) + X)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden=1024
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1,batch_first=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention=nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
    
    def forward(self, x):
        return self.attention(x,x,x)[0]


class MLP(nn.Module):
    """
    MLP is a simple implementaion of a feed-forward neural network(also known as a multi-layer perceptron)
    with two linear layers and a ReLU acivation function.
    """
    def __init__(self, n_state):
        super(MLP, self).__init__()
        self.fc1=nn.Linear(n_state, n_state)
        self.fc2=nn.Linear(n_state, n_state)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ResidualEncoderAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1,batch_first=True):
        super(ResidualEncoderAttentionBlock, self).__init__()
        self.attn=MultiHeadSelfAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.attn_ln=nn.LayerNorm(d_model)
        self.mlp=MLP(d_model)
        self.mlp_ln=nn.LayerNorm(d_model)

    def forward(self, x):
        x=x+self.attn(self.attn_ln(x))
        x=x+self.mlp(self.mlp_ln(x))
        return x

class FusionLayerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1,batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.text_self_attn =nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.text_add_norm_layer_1 = AddNorm(d_model, dropout=dropout)
        self.text_cross_attn_1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.text_add_norm_layer_2 = AddNorm(d_model, dropout=dropout)
        self.text_ffn = FeedForwardNetwork(d_model)
        self.text_add_norm_layer_3 = AddNorm(d_model, dropout=dropout)
        self.text_cross_attn_2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.text_add_norm_layer_4 = AddNorm(d_model, dropout=dropout)

        # 2.Decoder layer
        self.img_self_attn_2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.img_add_norm_layer_1 = AddNorm(d_model, dropout=dropout)
        self.img_cross_attn_1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.img_add_norm_layer_2 = AddNorm(d_model, dropout=dropout)
        self.img_ffn = FeedForwardNetwork(d_model)
        self.img_add_norm_layer_3 = AddNorm(d_model, dropout=dropout)
        self.img_cross_attn_2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.img_add_norm_layer_4 = AddNorm(d_model, dropout=dropout)

    def forward(self, pair):
        # self attention
        x1,x2 = pair
        y1= self.text_self_attn(x1)
        y1 = self.text_add_norm_layer_1(y1,x1)

         # self attention
        y2= self.img_self_attn_2(x2)
        y2 = self.img_add_norm_layer_1(y2,x2)

        # cross attention
        y1attn,_= self.text_cross_attn_1(y1,y2,y2)
        y1attn = self.text_add_norm_layer_2(y1attn,y1)

        y1attn1,_= self.text_cross_attn_2(y2,y1attn,y1attn)
        y1attn1 = self.text_add_norm_layer_4(y1attn1,y2)

        y1_after= self.text_ffn(y1attn1)
        y1_after = self.text_add_norm_layer_3(y1_after,y1attn1)


        # cross attention
        y2attn,_= self.img_cross_attn_1(y2,y1,y1)
        y2attn = self.img_add_norm_layer_2(y2attn,y2)

        y2attn1,_= self.img_cross_attn_2(y1,y2attn,y2attn)
        y2attn1 = self.img_add_norm_layer_4(y2attn1,y1)

        y2_after= self.img_ffn(y2attn1)
        y2_after = self.img_add_norm_layer_3(y2_after,y2attn1)

        return (y1_after,y2_after)

    
class FusionLayer(nn.Module):
    def __init__(self,d_model,num_layer,device,n_head=4,dropout=0.1):
        super(FusionLayer, self).__init__()
        self.device=device
        self.d_model=d_model
        self.num_layer=num_layer
    
        self.fusionlayer= nn.Sequential(*[FusionLayerBlock(d_model,n_head,dropout) for _ in range(num_layer)])
    
    def forward(self,x1,x2):
    
        return  self.fusionlayer((x1,x2))

class CosineSimilarity(nn.Module):
    def __init__(self):
        pass

    @staticmethod
    def forward(x1, x2,n,device):
        """
        Args:
            x1 (torch.Tensor): .
            x2 (torch.Tensor): Second input tensor.
        """
        batch_size = int(x1.size(0)/n)
        k1 = rearrange(x1,"(b n) l c -> n b (l c)",b=batch_size)
        k2 = rearrange(x2,"(b n) l c -> n b (l c)",b=batch_size)
        result = torch.mean(F.cosine_similarity(k1, k2, dim=-1),0)
        return result


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target,n=1):
        """
        Args:
            output1 (torch.Tensor): Feature embeddings for the first input.
            output2 (torch.Tensor): Feature embeddings for the second input.
            target (torch.Tensor): Binary label indicating whether the inputs are similar (1) or dissimilar (0).
        """
        distance =  CosineSimilarity.forward(output1, output2,n,device=output1.device)
        distance =  (distance +1) /2
 
        loss_contrastive = torch.mean((1 - target) * torch.pow(distance, 2) +
                                     (target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive
    
class DecoderLayerBlock(nn.Module):
    def __init__(self,d_model, n_heads=4, dropout=0.1,batch_first=True):
        super(DecoderLayerBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn =ResidualEncoderAttentionBlock(d_model, n_heads, dropout=dropout,batch_first=batch_first) 
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.image_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm2 = AddNorm(d_model, dropout=dropout)
        self.text_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm3 = AddNorm(d_model,  dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model)
        self.add_norm4 = AddNorm(d_model, dropout=dropout)
    def forward(self,pair):
        x,imgs_feat,text_feat =pair
        y,_= self.self_attn(x,x,x)
        y = self.add_norm1(y,x)

        yattn,_= self.image_cross_attn(y,imgs_feat,imgs_feat)
        yattn = self.add_norm2(yattn,y)

        yattn2,_= self.text_cross_attn(yattn,text_feat,text_feat)
        yattn2 = self.add_norm3(yattn2,yattn)

        y_after= self.ffn(yattn2)
        y_after = self.add_norm4(y_after,yattn2)

        return (y_after,imgs_feat,text_feat)
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_layer,device,n_head=4,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.device=device
        self.d_model=d_model
        self.num_layer=num_layer
        self.decoderlayer= nn.Sequential(*[DecoderLayerBlock(d_model,n_head,dropout) for _ in range(num_layer)])
    
    def forward(self,x,imgs_feat,text_feat):
        return self.decoderlayer((x,imgs_feat,text_feat))

class SingleAttention(nn.Module):
    def __init__(self,d_model,n_heads=4,batch_first=True,dropout=0.1):
        super(SingleAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.image_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm2 = AddNorm(d_model, dropout=dropout)
       
        self.ffn = FeedForwardNetwork(d_model)
        self.add_norm4 = AddNorm(d_model, dropout=dropout)
    def forward(self,x,y1):
        y,_= self.self_attn(x,x,x)
        y = self.add_norm1(y,x)

        yattn,_= self.image_cross_attn(y,y1,y1)
        yattn = self.add_norm2(yattn,y)

        y_after= self.ffn(yattn)

        return y_after

class Textual_Image_Model(nn.Module):
    def __init__(self, config):
        super(Textual_Image_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_dim = 768
        self.num_encoder_layer=config["NUM_LAYERS"][0]
        self.num_enrich_layer=config["NUM_LAYERS"][1]
        self.num_decoder_layer=config["NUM_LAYERS"][2]

        #image encoder
        self.image_processor =  AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.image_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.batch_norm2D = nn.BatchNorm2d(3, affine=False).to(self.device)
        
        #text encoder
        self.text_tokenizer =RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
        self.bert_model=RobertaModel.from_pretrained("FacebookAI/roberta-base")
        self.text_projection = nn.Linear(self.encoder_dim, self.encoder_dim)
        #Image  Fusion Attention
        local_reso = 8 * 8
        local_scale = local_reso ** -0.5
        self.position_embedding_image = nn.Parameter(local_scale * torch.randn(local_reso))
        self.position_embedding_text = build(self.encoder_dim)
        self.fusion_image_layer = FusionLayer(self.encoder_dim,self.num_encoder_layer,self.device)
        self.constrasive_loss = ContrastiveLoss()
        #enrich layer
        # self.enrich_layer = EnrichLayer(self.encoder_dim,self.num_enrich_layer,self.device)
        # self.enrich_text_layer = EnrichLayer(self.encoder_dim,self.num_text_layer,self.device)


        # Image Decoder Layer
        self.decoder_layer1 = DecoderLayer(self.encoder_dim,self.num_decoder_layer,self.device)
        # self.decoder_layer2 = DecoderLayer(self.encoder_dim,self.num_decoder_layer,self.device)

        # self.decoder= SingleAttention(self.encoder_dim)
        
        self.img_fc = self.get_img_fc()
        self.text_fc = self.get_text_fc()

    def get_img_fc(self, use_ln=True):
        # if use_ln:
        #     return nn.Sequential(
        #         nn.Linear(self.encoder_dim, 1024),
        #         nn.LayerNorm(1024, eps=1e-12),
        #     )
        # else:
        return nn.Linear(self.encoder_dim, 1024)

    def get_text_fc(self):
        return nn.Sequential(
            nn.Linear(self.encoder_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
        )
    def st_pooling(self, feat, bs):
        # spatial pooling
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [bt,c,l]->[bt,c]
        # temporal pooling
        feat = rearrange(feat, '(b t) c -> b c t', b=bs)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [b,c]
        # projection
        feat = self.img_fc(feat)
        return feat

    def ts_pooling(self, feat, bs):
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [bt,c,l]->[bt,c]
        # temporal pooling
        feat = rearrange(feat, '(b t) c -> b c t', b=bs)
        feat = F.adaptive_avg_pool1d(feat, 1).squeeze()  # [b,c]
        # projection
        feat = self.text_fc(feat)
        return feat
            
    def images_encoder(self,images):
        inputs = self.image_processor(images, return_tensors="pt",do_rescale=False).to(self.device)
        outputs= self.image_model(**inputs)
        return outputs.last_hidden_state
    
    def text_encoder(self,text):
        inputs = self.text_tokenizer.batch_encode_plus(text,max_length=64,padding="max_length",  return_special_tokens_mask=True, return_tensors="pt",  truncation=True).to(self.device)
        tokenizer_input = {k: inputs[k] for k in ['input_ids', 'attention_mask']}
        outputs = self.bert_model(**tokenizer_input)
        return outputs.last_hidden_state

    def forward(self,x):
        # x = {"local_images": PIL.Image[n],
        #      "sentences": List[m]}
        imgs= x['local_images']  #[b,n,c,h,w]
        texts = x['sentences']
        if self.training:
            labels = x['labels']
        b,n = imgs.size()[:2]
        m = len(texts)
        
        # 1. Image Encoder
        imgs = rearrange(imgs, 'b n c h w -> (b n) c h w') #[bn,c,h,w]
        # norm_imgs=self.batch_norm2D(imgs)
        # norm_imgs = (norm_imgs - torch.min(norm_imgs)) / (torch.max(norm_imgs) - torch.min(norm_imgs))
        imgs_feat=self.images_encoder(imgs) # [ bn, 64, 768]
        imgs_feat=imgs_feat/imgs_feat.norm(dim=-1, keepdim=True)
        # 2. Text Encoder
        texts_feat=self.text_encoder(texts) # [m,64,768]
        texts_feat=texts_feat/texts_feat.norm(dim=-1, keepdim=True)
        texts_feat = texts_feat.unsqueeze(0)
        texts_feat= texts_feat.repeat(n,1,1,1)
        texts_feat=rearrange(texts_feat, 'b n c h -> (b n) c h') 
        texts_feat = self.text_projection(texts_feat)
        check_hidden_feat = texts_feat.clone()

        imgs_feat = self.position_embedding_image + imgs_feat.permute(0,2,1)
        imgs_feat = imgs_feat.permute(0,2,1)
        # texts_feat = self.position_embedding_text(texts_feat)
        imgs_feat_clone = imgs_feat.clone()

        # 3. Enhance Image and Text Features
        imgs_feat,texts_feat = self.fusion_image_layer(imgs_feat,texts_feat)
        # 3.1 Enrich Layer
        # hidden_feat = self.enrich_layer(imgs_feat,texts_feat)
        # texts_feat = self.enrich_text_layer(texts_feat)

        # 4. Decoder Layer
        decoder_feats,_,_ = self.decoder_layer1(imgs_feat_clone,imgs_feat,texts_feat) 


        logits = CosineSimilarity.forward(check_hidden_feat, decoder_feats,device=self.device,n=n)

        # 4. Contrastive Loss
        if self.training:
            loss=self.constrasive_loss(texts_feat,decoder_feats,labels,n=n)
            return dict({"logits": logits,"loss":loss}  )
        else:
            return dict({"logits": logits})


def build_textual_image_model(config: dict):

    model = Textual_Image_Model(config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model
