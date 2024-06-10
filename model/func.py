
import torch.nn as nn
import torch
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = 1e-6
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.01):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, Y, X):
        return self.layer_norm(self.dropout(Y) + X)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden=1024
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.linear1(x).relu()))
    


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

        # 2.Decoder layer
        self.img_self_attn_2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.img_add_norm_layer_1 = AddNorm(d_model, dropout=dropout)
        self.img_cross_attn_1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first)
        self.img_add_norm_layer_2 = AddNorm(d_model, dropout=dropout)
        self.img_ffn = FeedForwardNetwork(d_model)
        self.img_add_norm_layer_3 = AddNorm(d_model, dropout=dropout)

    def forward(self, pair):
        # self attention
        x1,x2 = pair
        y1,_= self.text_self_attn(x1,x1,x1)
        y1 = self.text_add_norm_layer_1(y1,x1)

         # self attention
        y2,_= self.img_self_attn_2(x2,x2,x2)
        y2 = self.img_add_norm_layer_1(y2,x2)

        # cross attention
        y1attn,_= self.text_cross_attn_1(y1,y2,y2) 
        y1attn = self.text_add_norm_layer_2(y1attn,y1)

        y1_after= self.text_ffn(y1attn)
        y1_after = self.text_add_norm_layer_3(y1_after,y1attn)


        # cross attention
        y2attn,_= self.img_cross_attn_1(y2,y1,y1) 
        y2attn = self.img_add_norm_layer_2(y2attn,y2)


        y2_after= self.img_ffn(y2attn)
        y2_after = self.img_add_norm_layer_3(y2_after,y2attn)

        return (y1attn,y2attn)

    
class FusionLayer(nn.Module):
    def __init__(self,d_model,num_layer,device,n_head=4,dropout=0.1,batch_first=True):
        super(Ff).__init__()
        self.device=device
        self.d_model=d_model
        self.num_layer=num_layer
    
        self.fusionlayer= nn.Sequential(*[FusionLayerBlock(d_model,n_head,dropout,batch_first) for _ in range(num_layer)])
    
    def forward(self,x1,x2):
    
        return  self.fusionlayer((x1,x2))


class DecoderLayerBlock(nn.Module):
    def __init__(self,d_model, n_heads=4, dropout=0.1,batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1.Encoder layer
        self.self_attn =nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=batch_first) 
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
