import torch.nn as nn
import torch
from transformers import AutoImageProcessor, Swinv2Model, RobertaTokenizerFast, RobertaModel,  DeformableDetrModel
from utils.utils import distributed_rank
from einops import rearrange,repeat
from torch.nn import functional as F
from model.position_embedding import build

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.pos_encoder = build(d_model,dropout)

    def forward(self, x):
        batch_size, length, feat = x.shape
        x = x.permute(1, 0, 2)  # (length, batch_size, feat)
        x = self.pos_encoder(x)
        output, _ = self.multihead_attn(x, x, x)
        output = output.permute(1, 0, 2) +x.permute(1, 0, 2)   # (batch_size, length, feat) 
        
        return output 

class CrossModalAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)


        self.query_embed= build(q_dim,dropout)
        self.key_embed= build(k_dim,dropout)
        self.value_embed= build(v_dim,dropout)
        self.query_proj = nn.Linear(q_dim, q_dim)
        self.key_proj = nn.Linear(k_dim, q_dim)
        self.value_proj = nn.Linear(v_dim, v_dim)

        self.multihead_attn = nn.MultiheadAttention(q_dim, num_heads)
        self.layer_norm = nn.LayerNorm(q_dim)

    def forward(self, query, key, value):
        """
        Args:
            query (Tensor): Input tensor with shape (batch_size, q_len, q_dim)
            key (Tensor): Input tensor with shape (batch_size, k_len, k_dim)
            value (Tensor): Input tensor with shape (batch_size, v_len, v_dim)
        Returns:
            Tensor: Output tensor with shape (batch_size, q_len, q_dim)
        """
        batch_size, q_len, _ = query.size()
        _, k_len, _ = key.size()
        _, v_len, _ = value.size()

        # Project the query, key, and value tensors
        query_proj = self.query_proj(query)
        key_proj = self.key_proj(key)
        value_proj = self.value_proj(value)


        query = self.query_embed(query_proj)
        key = self.key_embed(key_proj)
        value = self.value_embed(value_proj)

        # Permute the tensors to match the expected input shape of nn.MultiheadAttention
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # Apply cross-modal attention
        attn_output, _ = self.multihead_attn(query, key, value)

        # Permute the output tensor back to (batch_size, q_len, q_dim)
        attn_output = attn_output.permute(1, 0, 2)

        # Apply layer normalization
        temp= attn_output + query_proj 
        output = self.dropout(self.layer_norm(temp))

        return output



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

        #Image  Fusion Attention
        self.image_self_attn = SelfAttention(self.encoder_dim).to(self.device)

        #Text Fusion Attention
        self.text_self_attn = SelfAttention(self.encoder_dim).to(self.device)

        # Image to Text Enhance
        self.image_text_attn = CrossModalAttention(self.encoder_dim,self.encoder_dim,self.encoder_dim).to(self.device)
        self.ffn= FeedForwardNetwork(self.encoder_dim).to(self.device)

        # Text to Image Enhance
        self.text_image_attn =  CrossModalAttention(self.encoder_dim,self.encoder_dim,self.encoder_dim).to(self.device)
        self.ffn2= FeedForwardNetwork(self.encoder_dim).to(self.device)


        # Decode Layer
        self.self_attn= SelfAttention(self.encoder_dim).to(self.device)
        self.image_cross_attn= CrossModalAttention(self.encoder_dim,self.encoder_dim,self.encoder_dim).to(self.device)
        self.text_cross_attn= CrossModalAttention(self.encoder_dim,self.encoder_dim,self.encoder_dim).to(self.device)
        self.ffn3= FeedForwardNetwork(self.encoder_dim).to(self.device)

        # Cosine Similarity
        self.alpha=nn.Parameter(torch.tensor(10.0),requires_grad=True)      

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
        norm_imgs=self.batch_norm2D(imgs)
        norm_imgs = (norm_imgs - torch.min(norm_imgs)) / (torch.max(norm_imgs) - torch.min(norm_imgs))
        imgs_feat=self.images_encoder(norm_imgs).requires_grad_() # [ bn, 64, 768]
        
        # 2. Text Encoder
        texts_feat=self.text_encoder(texts).requires_grad_() # [m,64,768]
        texts_feat=repeat(texts_feat, 'm l c -> (repeat m) l c', repeat=n)
        hidden_feat = texts_feat.clone()

        # 3. Image fusion Attention
        # 3.2 Self Attention
        imgs_feat = self.image_self_attn(imgs_feat)

        # 4. Text fusion Attention
        # 4.2 Self Attention
        texts_feat = self.text_self_attn(texts_feat)

        # 5. Enhance Image and Text
        # Image to text
        image_text_attn = self.image_text_attn(imgs_feat,texts_feat,texts_feat)
        enhance_image = self.ffn(image_text_attn)
        image_features=enhance_image.clone()

        # Text to Image

        text_image_attn = self.text_image_attn(texts_feat,imgs_feat,imgs_feat)
        enhance_text = self.ffn(text_image_attn)
        text_features=enhance_text.clone()


        # 6. overall fusion
        # 6.2 Self Attention
        cross_image_fusion = self.self_attn(hidden_feat)

        # 6.3 Image Cross Attention
        imagec_fusion = self.image_cross_attn(cross_image_fusion,image_features,image_features)

        # 6.4 Text Cross Attention
        textc_fusion = self.text_cross_attn(imagec_fusion,text_features,text_features)
        overall_fusion = self.ffn3(textc_fusion)

        # 7. Rearrange batch
        overall_fusion = rearrange(overall_fusion, '(b n) l c -> n b (l c)', b=b)
        hidden_feat = rearrange(hidden_feat, '(b m) l c -> b m (l c)',m=b)
        
        # 8. Cosine Similarity
        logits = F.cosine_similarity(overall_fusion, hidden_feat, dim=-1)
    
        logits=self.alpha*( torch.sum(logits,0)/logits.shape[0])
        return dict({"logits": logits}  )

def build_textual_image_model(config: dict):

    model = Textual_Image_Model(config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model
