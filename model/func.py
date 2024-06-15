
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
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
        self.layer_norm = nn.LayerNorm(d_model)

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
    
class CausalSelfAttention(nn.Module):

    def __init__(self, d_model,seq_length,n_head=4,dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.bias=True
        self.seq_length=seq_length
        assert d_model % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=self.bias)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=self.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.n_embd = d_model
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(seq_length, seq_length))
                                        .view(1, 1, seq_length, seq_length))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):

    def __init__(self, d_model,dropout=0.,bias=True):
        super().__init__()
        self.d_model = d_model
        self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
   
class Weird_Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "The hidden size is not a multiple of the number of attention heads"

        self.d_model = d_model
        self.n_heads = n_heads
        
        self.attention_head_size = int(d_model / n_heads)
        self.all_head_size = n_heads * self.attention_head_size

        # global_local block (glb)
        self.glb_q = nn.Linear(d_model, self.all_head_size)
        self.glb_k = nn.Linear(d_model, self.all_head_size)
        self.glb_v = nn.Linear(d_model, self.all_head_size)

        # prompt_local block (plb)
        self.plb_q = nn.Linear(d_model, self.all_head_size)
        self.plb_k = nn.Linear(d_model, self.all_head_size)
        self.plb_v = nn.Linear(d_model, self.all_head_size)

        self.dense = nn.Linear(d_model, d_model)
        self.glb_dropout = nn.Dropout(p=dropout)
        self.plb_dropout = nn.Dropout(p=dropout)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,global_feat,local_feat,text_feat,batch_first=False):
        '''
        global_feat: [Seq_length x Batch_size x Hidden_size]
        local_feat: [Seq_length x Batch_size x Hidden_size]
        text_feat: [Seq_length x Batch_size x Hidden_size]
        '''
        if batch_first == False:
            global_feat = global_feat.permute(1, 0, 2)  # [Batch_size x Seq_length x Hidden_size]
            local_feat = local_feat.permute(1, 0, 2)
            text_feat = text_feat.permute(1, 0, 2)

        # global_local block (glb)
        glb_mixed_q_layer = self.glb_q(global_feat)  # [Batch_size x Seq_length x Hidden_size]
        glb_mixed_k_layer = self.glb_k(local_feat)  # [Batch_size x Seq_length x Hidden_size]
        glb_mixed_v_layer = self.glb_v(local_feat)  # [Batch_size x Seq_length x Hidden_size]

        # prompt_local block (plb)
        plb_mixed_q_layer = self.plb_q(local_feat)
        plb_mixed_k_layer = self.plb_k(text_feat)
        plb_mixed_v_layer = self.plb_v(text_feat)


        # global_local block (glb)
        glb_q_layer = self.transpose_for_scores(
            glb_mixed_q_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        glb_k_layer = self.transpose_for_scores(glb_mixed_k_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        glb_v_layer = self.transpose_for_scores(
            glb_mixed_v_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        # prompt_local block (plb)
        plb_q_layer = self.transpose_for_scores(plb_mixed_q_layer)
        plb_k_layer = self.transpose_for_scores(plb_mixed_k_layer)
        plb_v_layer = self.transpose_for_scores(plb_mixed_v_layer)

        # global_local block (glb)
        glb_attention_scores = torch.matmul(glb_q_layer, glb_k_layer.transpose(-1,-2)) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        glb_attention_scores = glb_attention_scores / math.sqrt(self.attention_head_size) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        glb_attention_probs = nn.Softmax(dim=-1)(glb_attention_scores) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        glb_attention_probs = self.glb_dropout(glb_attention_probs) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        bridge_value=glb_attention_probs

        # prompt_local block (plb)
        plb_attention_scores = torch.matmul(plb_q_layer, plb_k_layer.transpose(-1,-2))
        plb_attention_scores = plb_attention_scores / math.sqrt(self.attention_head_size)
        plb_attention_probs = nn.Softmax(dim=-1)(plb_attention_scores)
        plb_attention_probs = self.plb_dropout(plb_attention_probs)

        plb_context_enhanced=torch.matmul(bridge_value,plb_attention_probs)

        # add value
        glb_context_layer = torch.matmul(glb_attention_probs, glb_v_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]
        plb_context_layer = torch.matmul(plb_context_enhanced, plb_v_layer)


        # glb context layer
        glb_context_layer = glb_context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        glb_new_context_layer_shape = glb_context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        glb_context_layer = glb_context_layer.view(*glb_new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]


        # plb context layer
        plb_context_layer = plb_context_layer.permute(0, 2, 1,
                                              3).contiguous()  # [Batch_size x Seq_length x Num_of_heads x Head_size]
        plb_new_context_layer_shape = plb_context_layer.size()[:-2] + (
        self.all_head_size,)  # [Batch_size x Seq_length x Hidden_size]
        plb_context_layer = plb_context_layer.view(*plb_new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        last_context_layer = glb_context_layer + plb_context_layer
        output = self.dense(last_context_layer)

        return output
