import torch.nn as nn
import torch
from transformers import AutoImageProcessor, Swinv2Model, RobertaTokenizerFast, RobertaModel,  DeformableDetrModel
from utils.utils import distributed_rank
from einops import rearrange,repeat
from torch.nn import functional as F
from model.position_embedding import build


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.layer_norm(self.dropout(Y) + X)

class MulNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MulNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.layer_norm(self.dropout(Y) * X)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (batch_size, sequence_length, d_model)
        Returns:
            Tensor: Output tensor with shape (batch_size, sequence_length, d_model)
        """
        batch_size, sequence_length, _ = x.size()

        # Apply the first linear layer and activation function
        ffn_output = self.linear1(x)
        ffn_output = self.activation(ffn_output)

        # Apply dropout
        ffn_output = self.dropout(ffn_output)

        # Apply the second linear layer
        ffn_output = self.linear2(ffn_output)

        return ffn_output

class EnrichBlock(nn.Module):
    def __init__(self,d_model, n_heads=8, dropout=0.1,batch_first=True):
        super(EnrichBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model)
        self.add_norm3 = AddNorm(d_model, dropout=dropout)
    
    def forward(self,x):
        y,_= self.self_attn(x,x,x)
        y = self.add_norm1(y,x)
        y_after= self.ffn(y)
        y_after = self.add_norm3(y_after,y)
        return y_after

class EnrichLayer(nn.Module):
    def __init__(self,d_model,num_layer,device,n_head=8,dropout=0.1):
        super(EnrichLayer, self).__init__()
        self.device=device
        self.d_model=d_model
        self.num_layer=num_layer
        self.enrichlayer= nn.ModuleList([EnrichBlock(d_model,n_head,dropout) for _ in range(num_layer)])
    
    def forward(self,x):
        for i in range(self.num_layer):
            x = self.enrichlayer[i](x)
        return x


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
        result = torch.tensor([],requires_grad=True,device=device)
        count=0
        for i in range(batch_size):
            temp=None
            for j in range(n):
                a= rearrange(x1[count],"l c -> (l c)")
                b= rearrange(x2[count],"l c -> (l c)")
                if temp is None:
                    temp = F.cosine_similarity(a, b, dim=-1)
                else:
                    temp = temp + F.cosine_similarity(a, b, dim=-1)
                count+=1
            p = temp/n
            result = torch.cat((result,p.unsqueeze(0)),0)
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
    def __init__(self,d_model, n_heads=8, dropout=0.1,batch_first=True):
        super(DecoderLayerBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.image_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm2 = AddNorm(d_model, dropout=dropout)
        self.text_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm3 = AddNorm(d_model,  dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model)
        self.add_norm4 = AddNorm(d_model, dropout=dropout)
    def forward(self,x,imgs_feat,text_feat):
        y,_= self.self_attn(x,x,x)
        y = self.add_norm1(y,x)

        yattn,_= self.image_cross_attn(y,imgs_feat,imgs_feat)
        yattn = self.add_norm2(yattn,y)

        yattn2,_= self.text_cross_attn(yattn,text_feat,text_feat)
        yattn2 = self.add_norm3(yattn2,yattn)

        y_after= self.ffn(yattn2)
        y_after = self.add_norm4(y_after,yattn2)

        return y_after 
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_layer,device,n_head=8,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.device=device
        self.d_model=d_model
        self.num_layer=num_layer
        self.decoderlayer= nn.ModuleList([DecoderLayerBlock(d_model,n_head,dropout) for _ in range(num_layer)])
    
    def forward(self,x,imgs_feat,text_feat):
        for i in range(self.num_layer):
            x = self.decoderlayer[i](x,imgs_feat,text_feat)
        return x

class EnrichModel(nn.Module):
    def __init__(self, config):
        super(EnrichModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_dim = 768
        self.num_image_layer=config["NUM_LAYERS"][0]
        self.num_text_layer=config["NUM_LAYERS"][1]
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
        self.position_embedding_image = build(self.encoder_dim)
        self.position_embedding_text = build(self.encoder_dim)
        self.enrich_image_layer = EnrichLayer(self.encoder_dim,self.num_image_layer,self.device)
        self.enrich_text_layer = EnrichLayer(self.encoder_dim,self.num_text_layer,self.device)

        self.constrasive_loss = ContrastiveLoss()

        # Image Decoder Layer
        self.decoder_layer = DecoderLayer(self.encoder_dim,self.num_decoder_layer,self.device)
        self.decoder_embedding =build(self.encoder_dim)
        
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
        imgs_feat=self.images_encoder(imgs).requires_grad_() # [ bn, 64, 768]
        hidden_feat = imgs_feat.clone()
        
        # 2. Text Encoder
        texts_feat=self.text_encoder(texts).requires_grad_() # [m,64,768]
        texts_feat = self.text_projection(texts_feat)
        texts_feat = texts_feat.unsqueeze(0)
        texts_feat= texts_feat.repeat(n,1,1,1)
        texts_feat=rearrange(texts_feat, 'b n c h -> (b n) c h') 
        check_hidden_feat = texts_feat.clone()

        hidden_feat = self.decoder_embedding(hidden_feat)
        imgs_feat = self.position_embedding_image(imgs_feat)
        texts_feat = self.position_embedding_text(texts_feat)
      
        # 3. Enhance Image and Text Features
        imgs_feat = self.enrich_image_layer(imgs_feat)
        texts_feat = self.enrich_text_layer(texts_feat)

        # 4. Decoder Layer
        decoder_feats = self.decoder_layer(hidden_feat,imgs_feat,texts_feat) 
        logits = CosineSimilarity.forward(check_hidden_feat, decoder_feats,device=self.device,n=n)

        # 4. Contrastive Loss
        if self.training:
            loss=self.constrasive_loss(texts_feat,decoder_feats,labels,n=n)
            return dict({"logits": logits,"loss":loss}  )
        else:
            return dict({"logits": logits})


def build_enrich(config: dict):

    model = EnrichModel(config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model
