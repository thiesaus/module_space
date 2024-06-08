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
class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


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
        # ffn_output = self.activation(ffn_output)

        # Apply dropout
        ffn_output = self.dropout(ffn_output)

        # Apply the second linear layer
        ffn_output = self.linear2(ffn_output)

        return ffn_output

class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class FusionLayerBlock(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.add_norm2 = AddNorm(d_model, dropout=dropout)
        self.add_norm3 = AddNorm(d_model, dropout=dropout)

        # 2.Decoder layer
        self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.add_norm4 = AddNorm(d_model, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
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
        y2attn = self.add_norm5(y2attn,y1)

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
    def forward(x1, x2,device):
        """
        Args:
            x1 (torch.Tensor): .
            x2 (torch.Tensor): Second input tensor.
        """
        batch_size,length, _ = x1.size()
        result = torch.zeros(batch_size,device=device)
        for i in range(batch_size):
            _x1 = rearrange(x1[i], 'l c -> (l c)')
            _x2 = rearrange(x2[i], 'l c -> (l c)')
            result[i] = F.cosine_similarity(_x1, _x2, dim=-1)
        
        return result


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        """
        Args:
            output1 (torch.Tensor): Feature embeddings for the first input.
            output2 (torch.Tensor): Feature embeddings for the second input.
            target (torch.Tensor): Binary label indicating whether the inputs are similar (1) or dissimilar (0).
        """
        distance =  CosineSimilarity.forward(output1, output2,device=output1.device)
        loss_contrastive = torch.mean((1 - target) * torch.pow(distance, 2) +
                                     (target) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive
    
class DecoderLayer(nn.Module):
    def __init__(self,d_model, n_heads=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.add_norm1 = AddNorm(d_model, dropout=dropout)
        self.image_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.add_norm2 = AddNorm(d_model, dropout=dropout)
        self.text_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
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

        return y_after * text_feat

class Textual_Image_Model(nn.Module):
    def __init__(self, config):
        super(Textual_Image_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_dim = 768

        #image encoder
        self.image_processor =  AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.image_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.batch_norm2D = nn.BatchNorm2d(3, affine=False).to(self.device)
        
        #text encoder
        self.text_tokenizer =RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
        self.bert_model=RobertaModel.from_pretrained("FacebookAI/roberta-base")
        self.text_projection = self.get_text_fc()
        #Image  Fusion Attention
        self.position_embedding_image = build(self.encoder_dim)
        self.position_embedding_text = build(self.encoder_dim)
        self.fusion_image_layer = FusionLayer(self.encoder_dim,config["NUM_LAYERS"],self.device)
        self.constrasive_loss = ContrastiveLoss()
        self.alpha = nn.Parameter(torch.tensor(10.0),requires_grad=True)

        # Image Decoder Layer
        self.decoder_layer = DecoderLayer(self.encoder_dim)
        self.decoder_embedding =build(self.encoder_dim)
        

    def get_img_fc(self, use_ln=True):
        if use_ln:
            return nn.Sequential(
                nn.Linear(self.encoder_dim, 1024),
                nn.LayerNorm(1024, eps=1e-12),
            )
        else:
            return nn.Linear(self.encoder_dim, 1024)

    def get_text_fc(self):
        return nn.Sequential(
            nn.Linear(self.encoder_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024,self.encoder_dim),
        )
            
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
        b,n = imgs.size()[:2]
        m = len(texts)
        
        # 1. Image Encoder
        imgs = rearrange(imgs, 'b n c h w -> (b n) c h w') #[bn,c,h,w]
        # norm_imgs=self.batch_norm2D(imgs)
        # norm_imgs = (norm_imgs - torch.min(norm_imgs)) / (torch.max(norm_imgs) - torch.min(norm_imgs))
        imgs_feat=self.images_encoder(imgs).requires_grad_() # [ bn, 64, 768]
        
        # 2. Text Encoder
        texts_feat=self.text_encoder(texts).requires_grad_() # [m,64,768]
        texts_feat=repeat(texts_feat, 'm l c -> (repeat m) l c', repeat=n)
        texts_feat = self.text_projection(texts_feat)
        check_hidden_feat = texts_feat.clone()

        imgs_feat = self.position_embedding_image(imgs_feat)
        texts_feat = self.position_embedding_text(texts_feat)
        hidden_feat = imgs_feat.clone()
        hidden_feat = self.decoder_embedding(hidden_feat)
        # 3. Enhance Image and Text Features
        imgs_feat,texts_feat = self.fusion_image_layer(imgs_feat.permute(1,0,2),texts_feat.permute(1,0,2))

        # 4. Decoder Layer
        decoder_feats = self.decoder_layer(hidden_feat.permute(1,0,2),imgs_feat,texts_feat) 

        # 4. Contrastive Loss
        loss=self.constrasive_loss(texts_feat.permute(1,0,2),decoder_feats.permute(1,0,2),torch.zeros(n*m).to(self.device) + 0.1)

        # 5. decoder Projection
        # decoder_feats = self.st_pooling(rearrange(decoder_feats,"l b c -> b c l"), b)
        # real_texts_feat = self.text_pooling(rearrange(real_texts_feat,"b l c -> b c l"), m)

        # 5. Cosine Similarity
        logits = CosineSimilarity.forward(check_hidden_feat, decoder_feats.permute(1,0,2),device=self.device)
        logits = logits.view(n,m)
        logits= torch.sum(logits,0)/logits.shape[0]

        return dict({"logits": logits,"loss":loss}  )

def build_textual_image_model(config: dict):

    model = Textual_Image_Model(config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model
