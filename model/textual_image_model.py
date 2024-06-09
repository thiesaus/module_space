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
        return self.layer_norm(Y + X)


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

        # # Apply dropout
        # ffn_output = self.dropout(ffn_output)

        # # Apply the second linear layer
        ffn_output = self.linear2(ffn_output)

        return ffn_output

class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = nn.functional.softmax
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Project the input into queries, keys, and values
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute the causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len,device=x.device)).unsqueeze(0).unsqueeze(0)

        # Compute the scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.attention(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute the output
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return out

class EnrichBlock(nn.Module):
    def __init__(self,d_model, n_heads=8, dropout=0.1,batch_first=True):
        super(EnrichBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm2 = AddNorm(d_model, dropout=dropout)
        self.cross_attn2= nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm3 = AddNorm(d_model, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model)
        self.add_norm4 = AddNorm(d_model, dropout=dropout)
    
    def forward(self,x_q,x1,x2):
        y= self.self_attn(x_q)
        y = self.add_norm1(y,x_q)
        yattn,_ =  self.cross_attn(y,x1,x1)
        yattn = self.add_norm2(yattn,y)
        yattn2,_ =  self.cross_attn(yattn,x2,x2)
        yattn2 = self.add_norm2(yattn2,yattn)
        y_after= self.ffn(yattn2)
        y_after = self.add_norm3(y_after,yattn2)
        return y_after

class EnrichLayer(nn.Module):
    def __init__(self,d_model,num_layer,device,n_head=8,dropout=0.1):
        super(EnrichLayer, self).__init__()
        self.device=device
        self.d_model=d_model
        self.num_layer=num_layer
        self.enrichlayer= nn.ModuleList([EnrichBlock(d_model,n_head,dropout) for _ in range(num_layer)])
    
    def forward(self,imgs_feat,text_feat):
        output = text_feat.clone()
        for i in range(self.num_layer):
            output = self.enrichlayer[i](output,imgs_feat,text_feat)
        return output
    

class FusionLayerBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1,batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm2 = AddNorm(d_model, dropout=dropout)
        self.add_norm3 = AddNorm(d_model, dropout=dropout)

        # 2.Decoder layer
        self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm4 = AddNorm(d_model, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.add_norm5 = AddNorm(d_model, dropout=dropout)
        self.add_norm6 = AddNorm(d_model, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model)
        self.ffn2 = FeedForwardNetwork(d_model)

    def forward(self, x1,x2):
        # self attention
        y1,_= self.self_attn(x1,x1,x1)
        y1 = self.add_norm1(y1,x1)

         # self attention
        y2,_= self.self_attn2(x2,x2,x2)
        y2 = self.add_norm4(y2,x2)

        # cross attention
        y1attn,_= self.cross_attn(y1,y2,y2)
        y1attn = self.add_norm2(y1attn,y1)

        y1_after= self.ffn(y1attn)
        y1_after = self.add_norm3(y1_after,y1attn)


        # cross attention
        y2attn,_= self.cross_attn2(y2,y1,y1)
        y2attn = self.add_norm5(y2attn,y2)

        y2_after= self.ffn2(y2attn)
        y2_after = self.add_norm6(y2_after,y2attn)

        return y1_after,y2_after


class FusionLayer(nn.Module):
    def __init__(self,d_model,num_layer,device,n_head=8,dropout=0.1):
        super(FusionLayer, self).__init__()
        self.device=device
        self.d_model=d_model
        self.num_layer=num_layer
    
        self.fusionlayer= nn.ModuleList([FusionLayerBlock(d_model,n_head,dropout) for _ in range(num_layer)])
    
    def forward(self,x1,x2):
        for i in range(self.num_layer):
            x1,x2 = self.fusionlayer[i](x1,x2)
        return x1,x2

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

class SingleAttention(nn.Module):
    def __init__(self,d_model,n_heads=8,batch_first=True,dropout=0.1):
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
        y_after = self.add_norm4(y_after,yattn)

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
        self.position_embedding_image = build(self.encoder_dim)
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
        imgs_feat=self.images_encoder(imgs).requires_grad_() # [ bn, 64, 768]
        imgs_feat=imgs_feat/imgs_feat.norm(dim=-1, keepdim=True)
        # 2. Text Encoder
        texts_feat=self.text_encoder(texts).requires_grad_() # [m,64,768]
        texts_feat=texts_feat/texts_feat.norm(dim=-1, keepdim=True)
        texts_feat = self.text_projection(texts_feat)
        texts_feat = texts_feat.unsqueeze(0)
        texts_feat= texts_feat.repeat(n,1,1,1)
        texts_feat=rearrange(texts_feat, 'b n c h -> (b n) c h') 
        check_hidden_feat = texts_feat.clone()

        imgs_feat = self.position_embedding_image(imgs_feat)
        texts_feat = self.position_embedding_text(texts_feat)
        imgs_feat_clone = imgs_feat.clone()
        texts_feat_clone = texts_feat.clone()
        # 3. Enhance Image and Text Features
        imgs_feat,texts_feat = self.fusion_image_layer(imgs_feat,texts_feat)
        # 3.1 Enrich Layer
        # hidden_feat = self.enrich_layer(imgs_feat,texts_feat)
        # texts_feat = self.enrich_text_layer(texts_feat)

        # 4. Decoder Layer
        decoder_feats = self.decoder_layer1(imgs_feat_clone,imgs_feat,texts_feat) 
        # decoder_feats_texts = self.decoder_layer2(texts_feat_clone,imgs_feat,texts_feat)

        # 5. Decoder
        # decoder_feats = self.decoder(decoder_feats_images,decoder_feats_texts) * texts_feat


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
