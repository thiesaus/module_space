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


class Model4(nn.Module):
    def __init__(self, ):
        super(Model4, self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        self.swinv2_model =Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        self.bert_model=  RobertaModel.from_pretrained("FacebookAI/roberta-base").to(self.device)
        self._freeze_encoder()

        self.img_dim = 256
        self.image_crop=4
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
        # self.reprocess_text1=make_layers(384, 192, 2, is_downsample=True)
       

        # process image
        # self.process_reso=32*32
        # self.process_scale = self.process_reso ** -0.5
        # self.process_image_linear = nn.Linear(self.img_dim,self.img_dim ).to(self.device)
        # self.process_emb_position=nn.Parameter(self.process_scale * randn(  self.process_reso)).to(self.device)



        # text_local
        self.textual_local_linear =nn.Linear(self.text_dim, self.img_dim).to(self.device)
        local_reso = 16* 16
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso)).to(self.device)
        self.fusion_text_local = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0.,
        ).to(self.device)

        # #text_global
        self.textual_global_linear = nn.Linear(self.text_dim, self.img_dim).to(self.device)
        global_reso =16* 16
        global_scale = global_reso ** -0.5
        self.pos_emb_global = nn.Parameter(global_scale * randn(global_reso)).to(self.device)
        self.fusion_text_global = nn.MultiheadAttention(
                embed_dim=self.img_dim,
                num_heads=4,
                dropout=0,
            ).to(self.device)

        # #full_fusion
         
        self.full_fusion_layer = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0,
        ).to(self.device)

        # #repeat_text
        self.repeat_linear = nn.Linear(self.text_dim, self.img_dim).to(self.device)

        self.repeat_text_layer = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0,
        ).to(self.device)


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
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        outputs = self.swinv2_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states 

    def forward(self, x):
        batch_feats=[self.processing_input(i) for i in x] # arr([{"local_images":PIL.Image[n],"global_images":1,"sentences":str[m]}])
        #example

        quantities=[batch["local_images"].shape[0] for batch in batch_feats] # arr([n1,n2,n3,...])
        norm_feats={k:torch.vstack([i[k] for i in batch_feats]) for k in batch_feats[0].keys()} 

        #fusion local_global
        _local_feat=norm_feats["local_images"].requires_grad_()
        _global_feat=norm_feats["global_images"].requires_grad_()
        text_feat=norm_feats["sentences"].requires_grad_()

        _local_feat=rearrange(_local_feat,"b (h w) c -> b c h w",h=8)
        _global_feat=rearrange(_global_feat,"b (h w) c -> b c h w",h=8)

        local_feat=self.cnn_image(_local_feat)
        global_feat=self.cnn_image(_global_feat)

        local_feat=rearrange(local_feat,"b c h w -> b (h w c)")
        global_feat=rearrange(global_feat,"b c h w -> b (h w c)")

        text_local_feat=self.textual_local(local_feat, text_feat)
        text_global_feat=self.textual_global(global_feat, text_feat)


        full_fusion_feat=self.full_fusion(text_global_feat, text_local_feat)
        
        full_feat=self.repeat_text(text_feat, full_fusion_feat)


        all_logits=[]
        for i,quantity in enumerate(quantities):
            ff=full_feat[i*quantity:(i+1)*quantity] if i+1<len(quantities) else full_feat[i*quantity:]
            lf=local_feat[i*quantity:(i+1)*quantity] if i+1<len(quantities) else local_feat[i*quantity:]
            logits=[]
            
            for fs_f in ff:
                logit=[]
                for text_f in lf:
                    logit.append(F.cosine_similarity(fs_f, text_f,dim=-1).item())
                logits.append(logit)
            all_logits.append( torch.Tensor(logits).to(self.device))

       
        
        return dict({"logits": all_logits}  )
    



    def processing_input(self,x):
        global_image = x["global_image"]
        global_image = self.process_image(global_image)
        global_image = global_image / global_image.norm(dim=-1, keepdim=True)
        global_image = global_image.repeat(len(x["local_images"]),1,1)
        
        processed={
            "local_images": self.process_image(x["local_images"]),
            # "global_images":torch.vstack([ self.preprocess(x["global_image"]) for _ in x["local_images"]]).view(-1,3,224,224).to(self.device),
            "sentences":self.process_text(x["sentences"])
        }

        feats={
            "local_images":processed["local_images"],
            # "global_images":self.clip.encode_image(processed["global_images"]),
            "sentences":processed["sentences"]
        }
        norm_feats={
            "local_images":feats["local_images"]/feats["local_images"].norm(dim=-1, keepdim=True),
            # "global_images":feats["global_images"]/feats["global_images"].norm(dim=-1, keepdim=True),
            "global_images":global_image,
            "sentences":feats["sentences"]/feats["sentences"].norm(dim=-1, keepdim=True)
        }
   
        return norm_feats
    

    def textual_local(self, local_image, text_feat):
        text_feat = self.textual_local_linear(text_feat)
        local_feat=local_image + self.pos_emb_local
        fusion_feat= self.fusion_text_local(
            query=text_feat,
            key=local_feat,
            value=local_feat
        )
        return fusion_feat[0] + text_feat
    
    def textual_global(self, global_image, text_feat):
        text_feat = self.textual_global_linear(text_feat)
        global_feat=global_image + self.pos_emb_global
        fusion_feat= self.fusion_text_global(
            query=text_feat,
            key=global_feat,
            value=global_feat
        )
        return fusion_feat[0] + text_feat
   
    def full_fusion(self,gt_feat,lt_feat):
        fusion_feat= self.full_fusion_layer(
            query=lt_feat,
            key=gt_feat,
            value=gt_feat
        )
        return fusion_feat[0] + lt_feat
    
    def repeat_text(self, text_feat, full_feat):
        text_feat = self.repeat_linear(text_feat)
        fusion_feat= self.repeat_text_layer(
            query=text_feat,
            key=full_feat,
            value=full_feat
        )
        return fusion_feat[0] *  text_feat
    

def build_model4(config: dict):



    model = Model4()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model