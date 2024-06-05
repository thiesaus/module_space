import torch.nn as nn
import torch
from transformers import AutoImageProcessor, Swinv2Model, RobertaTokenizerFast, RobertaModel,  DeformableDetrModel
from utils.utils import distributed_rank
from einops import rearrange,repeat
from torch.nn import functional as F
from model.position_embedding import build

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
        self.image_pos_embed = build(self.encoder_dim).to(self.device)
        self.image_self_attn = nn.MultiheadAttention(self.encoder_dim, num_heads=8, dropout=0.1, batch_first=True).to(self.device)

        #Text Fusion Attention
        self.text_pos_embed = build(self.encoder_dim).to(self.device)
        self.text_self_attn = nn.MultiheadAttention(self.encoder_dim, num_heads=8, dropout=0.1, batch_first=True).to(self.device)

        # Image to Text Enhance
        self.text_proj = nn.Linear(self.encoder_dim, self.encoder_dim).to(self.device)
        self.image_text_attn = nn.MultiheadAttention(self.encoder_dim, num_heads=8, dropout=0.1, batch_first=True).to(self.device)
        self.image_text_enhance_text_embed=build(self.encoder_dim).to(self.device)
        self.image_text_enhance_image_embed=build(self.encoder_dim).to(self.device)
        self.ffn= nn.Linear(self.encoder_dim, self.encoder_dim).to(self.device)

        # Text to Image Enhance
        self.text_proj_2 = nn.Linear(self.encoder_dim, self.encoder_dim).to(self.device)
        self.text_image_attn = nn.MultiheadAttention(self.encoder_dim, num_heads=8, dropout=0.1, batch_first=True).to(self.device)
        self.text_image_enhance_text_embed=build(self.encoder_dim).to(self.device)
        self.text_image_enhance_image_embed=build(self.encoder_dim).to(self.device)
        self.ffn2= nn.Linear(self.encoder_dim, self.encoder_dim).to(self.device)


        # Decode Layer
        self.text_proj_3 = nn.Linear(self.encoder_dim, self.encoder_dim).to(self.device)
        self.self_attn= nn.MultiheadAttention(self.encoder_dim, num_heads=8, dropout=0.1, batch_first=True).to(self.device)
        self.image_cross_attn= nn.MultiheadAttention(self.encoder_dim, num_heads=8, dropout=0.1, batch_first=True).to(self.device)
        self.text_cross_attn= nn.MultiheadAttention(self.encoder_dim, num_heads=8, dropout=0.1, batch_first=True).to(self.device)
        self.ffn3= nn.Linear(self.encoder_dim, self.encoder_dim).to(self.device)
        self.self_decode_embed=build(self.encoder_dim).to(self.device)
        self.image_cross_q_embed=build(self.encoder_dim).to(self.device)
        self.image_cross_kv_embed=build(self.encoder_dim).to(self.device)
        self.text_cross_q_embed=build(self.encoder_dim).to(self.device)
        self.text_cross_kv_embed=build(self.encoder_dim).to(self.device)

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
        hidden_feat = texts_feat.clone()

        # 3. Image fusion Attention
        # 3.1 Add embedding image
        imgs_feat = self.image_pos_embed(imgs_feat)
        # 3.2 Self Attention
        imgs_feat = self.image_self_attn(imgs_feat,imgs_feat,imgs_feat)[0]

        # 4. Text fusion Attention
        # 4.1 Add embedding text
        texts_feat = self.text_pos_embed(texts_feat)
        # 4.2 Self Attention
        texts_feat = self.text_self_attn(texts_feat,texts_feat,texts_feat)[0]

        # 5. Enhance Image and Text
        # Image to text
        text_proj = self.text_proj(imgs_feat)
        image_q =self.image_text_enhance_image_embed(imgs_feat)
        text_kv =self.image_text_enhance_text_embed(text_proj)
        image_text_attn = self.image_text_attn(image_q,text_kv,text_kv)[0] +image_q
        enhance_image = self.ffn(image_text_attn)
        image_features=enhance_image.clone()

        # Text to Image
        image_proj = self.text_proj_2(texts_feat)
        text_q =self.text_image_enhance_text_embed(texts_feat)
        image_kv = self.text_image_enhance_image_embed(image_proj)
        text_image_attn = self.text_image_attn(text_q,image_kv,image_kv)[0] +text_q
        enhance_text = self.ffn(text_image_attn)
        text_features=enhance_text.clone()


        # 6. overall fusion
        cross_image_q = repeat(hidden_feat.clone(), 'm l c -> (repeat m) l c', repeat=n)

        cross_image_q =  self.text_proj_3(cross_image_q)
        # 6.1 Add embedding image
        cross_image_q = self.self_decode_embed(cross_image_q)
        # 6.2 Self Attention
        cross_image_kv = image_features

        cross_image_fusion = self.self_attn(cross_image_q,cross_image_kv,cross_image_kv)[0] 

        # 6.3 Image Cross Attention
        imagec_q = self.image_cross_q_embed(cross_image_fusion)
        imagec_kv = self.image_cross_kv_embed(image_features)
        imagec_fusion = self.image_cross_attn(imagec_q,imagec_kv,imagec_kv)[0] + imagec_q

        # 6.4 Text Cross Attention
        textc_kv = repeat(text_features, 'm l c -> (repeat m) l c', repeat=n)
        textc_q = self.text_cross_q_embed(imagec_fusion)
        textc_kv = self.text_cross_kv_embed(textc_kv)
        textc_fusion = self.text_cross_attn(textc_q,textc_kv,textc_kv)[0] *  textc_q
        overall_fusion = self.ffn3(textc_fusion)

        # 7. Rearrange batch
        overall_fusion = rearrange(overall_fusion, '(b n) l c -> n b (l c)', b=b)
        hidden_feat = rearrange(hidden_feat, 'm l c -> m (l c)')
        
        # 8. Cosine Similarity
        logits = F.cosine_similarity(overall_fusion, hidden_feat, dim=-1)
    
        logits=torch.sum(logits,0)/logits.shape[0]
        return dict({"logits": logits}  )

def build_textual_image_model(config: dict):

    model = Textual_Image_Model(config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model
