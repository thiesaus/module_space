import torch
from torch import nn,randn
import numpy as np
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

class FeedForwardBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ln1 = nn.Linear(dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, dim)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        h_res = x
        x = self.norm(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.ln2(x)
        return x + h_res

class FusionBlock(nn.Module):
    def __init__(self,img_dim, text_dim,num_heads=4,device="cuda"):
        super(FusionBlock, self).__init__()
        self.device=device
        self.img_dim = img_dim
        self.text_dim = text_dim    
        local_reso = 16* 16
        local_scale = local_reso ** -0.5
        # self.emb = nn.Parameter(local_scale * randn(local_reso),requires_grad=True).to(self.device)
        # self.emb2 = nn.Parameter(local_scale * randn(local_reso),requires_grad=True).to(self.device)
        self.linear1 = nn.Linear(self.text_dim, self.img_dim).to(self.device)
        self.linear2 = nn.Linear(self.text_dim, self.img_dim).to(self.device)
        self.linear3 = nn.Linear(self.text_dim, self.img_dim).to(self.device)
        self.fusion = nn.MultiheadAttention(
        embed_dim=self.img_dim,
        num_heads=num_heads,
        dropout=0.,
        ).to(self.device)
        self.self_attn = nn.MultiheadAttention(
        embed_dim=self.img_dim,
        num_heads=num_heads,
        dropout=0.,
        ).to(self.device)
    
    def forward(self, query, key,value,is_mul=False):
        _query = self.linear1(query) 
        _key = self.linear2(key) 
        _value = self.linear3(value)
        fusion_feat = self.fusion(
            query=_query,
            key=_key,
            value=_value
        )
        fusion_feat2 = fusion_feat[0]
        fusion_feat3= self.self_attn(fusion_feat2, fusion_feat2, fusion_feat2)[0]

        if is_mul:
            return fusion_feat3.mul(query)
        else:
            return fusion_feat3.add(query)

class ZicZacBlock(nn.Module):
    def __init__(self,img_dim, text_dim,num_heads=4,is_last=False,device="cuda"):
        super(ZicZacBlock, self).__init__()
        self.device=device
        self.img_dim = img_dim
        self.text_dim = text_dim    
        self.layer1= FusionBlock(self.img_dim, self.text_dim, num_heads,device=self.device)
        self.layer2= FusionBlock(self.img_dim, self.text_dim, num_heads,device=self.device)
        self.layer3= FusionBlock(self.img_dim, self.text_dim, num_heads,device=self.device)
        self.layer4= FusionBlock(self.img_dim, self.text_dim, num_heads,device=self.device)
        self.fusion = nn.MultiheadAttention(
        embed_dim=self.img_dim,
        num_heads=num_heads,
        dropout=0.,
        ).to(self.device)
        self.is_last=is_last

    def forward(self, one, two):
        # if res is not None:
        #     fusion_1_1=self.layer1(query, res,is_add=True)
        #     if self.is_last:
        #         return query,key,self.layer2(key, fusion_1_1,is_mul=True)
        #     fusion_1_2=self.layer2(key, fusion_1_1,is_add=True)
        #     return query,key, fusion_1_2
        
        fusion_1=self.layer1(one, two,two,is_mul=self.is_last)
        fusion_2=self.layer2(two,one,one,is_mul=self.is_last)

        fusion_3=self.layer3(fusion_2, fusion_1,fusion_1,is_mul=self.is_last)
        fusion_4=self.layer4(fusion_2, fusion_1,fusion_1,is_mul=self.is_last)
        if self.is_last:
            return self.fusion(fusion_3, fusion_4, fusion_4)[0],None
        return fusion_3,fusion_4
       

def make_ziczac_layers(img_dim, text_dim, repeat_times,device="cuda"):
    blocks = []
    for i in range(repeat_times):
        blocks += [ZicZacBlock(img_dim, text_dim,device=device), ]
    blocks+=[ZicZacBlock(img_dim, text_dim,is_last=True,device=device),]
    return blocks

class Model4(nn.Module):
    def __init__(self, ):
        super(Model4, self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swinv2_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.bert_model=  RobertaModel.from_pretrained("FacebookAI/roberta-base").to(self.device)
        self._freeze_encoder()

        self.feature_dim=1024

        self.img_dim = 256
        self.text_dim = 256
        #reprocess image
        self.reprocess_image=make_layers(1024, 512, 2, is_downsample=False)
        self.reprocess_image1=make_layers(512, 512, 2, is_downsample=True)
        self.reprocess_image2=make_layers(512, 256, 2, is_downsample=True)
        self.reprocess_image3=make_layers(256 , 256 , 2, is_downsample=True)

        #reprocess text
        self.text_linear = nn.Linear(768, 384).to(self.device)
        self.text_linear1 = nn.Linear(384, 192).to(self.device)
        self.text_linear2 = nn.Linear(192, 96).to(self.device)
        self.text_linear3 = nn.Linear(96, 64).to(self.device)
        self.text_linear4 = nn.Linear(4096, 2048).to(self.device)
        self.text_linear5 = nn.Linear(2048, 1024).to(self.device)
        self.text_linear6 = nn.Linear(1024, 512).to(self.device)
        self.text_linear7 = nn.Linear(512, 256).to(self.device)
        self.fusion_fc = nn.Linear(self.text_dim, self.img_dim)
        # self.reprocess_text1=make_layers(384, 192, 2, is_downsample=True)
        self.numlayers=4
        self.supa_layer=make_ziczac_layers(self.img_dim, self.text_dim, self.numlayers,device=self.device)

 
        self.logit_scale = nn.Parameter(torch.tensor(10.0),requires_grad=True).to(self.device)
        local_reso = 8* 8
        local_scale = local_reso ** -0.5
        self.emb = nn.Parameter(local_scale * randn(local_reso),requires_grad=True).to(self.device)
        self.emb2 = nn.Parameter(local_scale * randn(local_reso),requires_grad=True).to(self.device)
    
    def forward(self, x):
        b, t = x["local_images"].size()[:2]
        batch_feats=self.processing_input(x)# arr([{"local_images":PIL.Image[n],"global_images":1,"sentences":str[m]}])
        #example

        # quantities=[batch["local_images"].shape[0] for batch in batch_feats] # arr([n1,n2,n3,...])
        # norm_feats={k:torch.vstack([i[k] for i in batch_feats]) for k in batch_feats[0].keys()} 

        #fusion local_global
        _local_feat=batch_feats["local_images"].requires_grad_()
        _local_feat=rearrange(_local_feat,"b a c -> b c a")
        _local_feat=_local_feat + self.emb
        _local_feat=rearrange(_local_feat,"b a c -> b c a")

        # _global_feat=norm_feats["global_images"].requires_grad_()
        # _local_feat = local_feat + self.emb

        _local_feat=rearrange(_local_feat,"b (h w) c -> b c h w",h=8)
        # _global_feat=rearrange(_global_feat,"b (h w) c -> b c h w",h=8)

        local_feat=self.cnn_image(_local_feat)

       
        # global_feat=self.cnn_image(_global_feat)

        local_feat=rearrange(local_feat,"b c h w -> b (h w c)")
        # global_feat=rearrange(global_feat,"b c h w -> b (h w c)")
        text_feat=batch_feats["sentences"].requires_grad_()
        text_feat = text_feat.unsqueeze(1)  # [b,c]->[b,1,c]
        text_feat = text_feat.repeat([1, t, 1])
        text_feat = rearrange(text_feat, 'b t c -> (b t) c')
        text_hidden = text_feat
        text_feat = self.fusion_fc(text_feat)
        # text_feat = rearrange(text_feat, 'bt c -> l bt c')

        full_feat=None
        local_feat_1= local_feat
        text_feat_1= text_feat
        for block in self.supa_layer:
            local_feat_1,text_feat_1 = block(local_feat_1,text_feat_1)
        full_feat=rearrange(local_feat_1,"(b t) c -> t b c",t=t)
        text_hidden = rearrange(text_hidden, '(b t) c -> t b c', t=t)
        logits = F.cosine_similarity(full_feat, text_hidden,dim=-1)
        temp=torch.zeros(logits.shape[1],device=self.device)
        for i in range(logits.shape[0]):
            temp= temp+logits[i]
        logits=temp/logits.shape[0]
        return dict({"logits": logits}  )
    

    def _freeze_encoder(self):
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

    def cnn_image(self, local_image):
        local_image = self.reprocess_image(local_image)
        local_image = self.reprocess_image1(local_image)
        local_image = self.reprocess_image2(local_image)
        local_image = self.reprocess_image3(local_image)
        return local_image
    
    def cnn_text(self, text):
        text = self.reprocess_text(text)

        # text = self.reprocess_text1(text)
        return text
    
    def process_image(self,image):
        #cut image to 8
        # width, height = image.size

        # col=1
        # row=2
        # # Calculate the dimensions of each sub-image
        # sub_width = width // col
        # sub_height = height // row

        # # Create a list to store the sub-images
        # sub_images = []

        # # Crop the image into 8 sub-images
        # for x in range(row):
        #     for y in range(col):
        #         left = x * sub_width
        #         top = y * sub_height
        #         right = left + sub_width
        #         bottom = top + sub_height
        #         sub_image = image.crop((left, top, right, bottom))
        #         sub_images.append(sub_image)
        temp=self.image_encoder(image)
        padded_temp = F.pad(temp, (0, 256, 0, 0, 0, 0), mode='constant', value=0)
        # processed_images=[ i + self.process_emb_position for i in temp]
        # linear_images=[self.process_image_linear(i) for i in temp]
        return padded_temp
    
    def process_text(self,texts):
        text=self.text_encoder(texts)
        text = self.text_linear(text)
        text = self.text_linear1(text)
        text = self.text_linear2(text)
        text = self.text_linear3(text)
        text = rearrange(text,"b w c -> b (w c)")
        text = self.text_linear4(text)
        text = self.text_linear5(text)
        text = self.text_linear6(text)
        text = self.text_linear7(text)
        return text

    def text_encoder(self, text):  # [1,3,768]
        inputs = self.tokenizer.batch_encode_plus(text,max_length=64,padding="max_length",  return_special_tokens_mask=True, return_tensors="pt",  truncation=True).to(self.device)
        tokenizer_input = {"input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"]}

        outputs = self.bert_model(**tokenizer_input)
        return outputs.last_hidden_state
    
    def image_encoder(self, image): # [1,49,768]
        inputs = self.image_processor(image, return_tensors="pt",do_rescale=False).to(self.device)
        outputs = self.swinv2_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states 


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
        text_hidden = self.text_linear6(text_hidden)
        text_hidden = self.text_linear7(text_hidden)
        if self.training:
            return text_hidden
        else:
            return  F.normalize(text_hidden, p=2, dim=-1)

    def processing_input(self,x):
        # global_image = x["global_image"]
        # global_image = self.process_image(global_image)
        # global_image = global_image / global_image.norm(dim=-1, keepdim=True)
        # sentences = [x["sentences"][0] for _ in range(len(x["local_images"]))]
        b, t = x['local_images'].size()[:2]
        local_img = rearrange(x['local_images'], 'b t c h w -> (b t) c h w')
        local_img=(local_img-  local_img.min())/ (local_img.max() - local_img.min())
        local_feat =  self.process_image(local_img)
        feats={
            "local_images": local_feat,
            # "global_images":torch.vstack([ self.preprocess(x["global_image"]) for _ in x["local_images"]]).view(-1,3,224,224).to(self.device),
            "sentences":self.textual_encoding( x['sentences'])
        }

        norm_feats={
            "local_images":feats["local_images"]/feats["local_images"].norm(dim=-1, keepdim=True),
            # "global_images":feats["global_images"]/feats["global_images"].norm(dim=-1, keepdim=True),
            # "global_images":global_image,
            "sentences":feats["sentences"]/feats["sentences"].norm(dim=-1, keepdim=True)
        }
   
        return norm_feats
    

def build_model4(config: dict):

    model = Model4()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model