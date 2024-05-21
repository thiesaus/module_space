import torch
from torch import nn,randn
import clip
import numpy as np
from model.position_embedding import build as build_position_embedding
import torch.nn.functional as F
from utils.utils import distributed_rank

class Model4(nn.Module):
    def __init__(self, ):
        super(Model4, self).__init__()
        model, preprocess = clip.load("RN50")
        self.clip=model
        self.preprocess=preprocess
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._freeze_encoder()

       
        self.img_dim = 1024
        self.image_crop=4
        self.text_dim = 1024
      
       

        # process image
        self.process_reso=32*32
        self.process_scale = self.process_reso ** -0.5
        self.process_image_linear = nn.Linear(self.img_dim,128 ).to(self.device)
        self.process_emb_position=nn.Parameter(self.process_scale * randn(  self.process_reso)).to(self.device)

        # text_local
        self.textual_local_linear = nn.Linear(self.text_dim, self.img_dim).to(self.device)
        local_reso = 32 * 32
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso)).to(self.device)
        self.fusion_text_local = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0.,
        ).to(self.device)

        #text_global
        self.textual_global_linear = nn.Linear(self.text_dim, self.img_dim).to(self.device)
        global_reso = 32 * 32
        global_scale = global_reso ** -0.5
        self.pos_emb_global = nn.Parameter(global_scale * randn(global_reso)).to(self.device)
        self.fusion_text_global = nn.MultiheadAttention(
                embed_dim=self.img_dim,
                num_heads=4,
                dropout=0,
            ).to(self.device)

        #full_fusion
         
        self.full_fusion_layer = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0,
        ).to(self.device)

        #repeat_text
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
        for p in list(self.clip.transformer.parameters()) + \
                 list(self.clip.ln_final.parameters()) + \
                 [self.clip.text_projection, ]:
            p.requires_grad = False

    def forward(self, x):
        batch_feats=[self.processing_input(i) for i in x] # arr([{"local_images":PIL.Image[n],"global_images":1,"sentences":str[m]}])
        #example

        quantities=[batch["local_images"].shape[0] for batch in batch_feats] # arr([n1,n2,n3,...])
        norm_feats={k:torch.vstack([i[k] for i in batch_feats]) for k in batch_feats[0].keys()} 

        #fusion local_global
        local_feat=norm_feats["local_images"]
        global_feat=norm_feats["global_images"]
        text_feat=norm_feats["sentences"]

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
    

    def process_image(self,image):
        #cut image to 8
        width, height = image.size

        # Calculate the dimensions of each sub-image
        sub_width = width // 2
        sub_height = height // 4

        # Create a list to store the sub-images
        sub_images = []

        # Crop the image into 8 sub-images
        for x in range(2):
            for y in range(4):
                left = x * sub_width
                top = y * sub_height
                right = left + sub_width
                bottom = top + sub_height
                sub_image = image.crop((left, top, right, bottom))
                sub_images.append(sub_image)
             
        processed_images=[self.clip.encode_image(self.preprocess(i).to(self.device).unsqueeze(0)).float() + 
                          self.process_emb_position for i in sub_images]
        linear_images=[self.process_image_linear(i) for i in processed_images]
        concatened_images=torch.hstack(linear_images).requires_grad_()
        return concatened_images



    def processing_input(self,x):
        global_image = x["global_image"]
        global_image = self.preprocess(global_image).unsqueeze(0).to(self.device)
        global_image = self.clip.encode_image(global_image).float()
        global_image = global_image / global_image.norm(dim=-1, keepdim=True)
        global_image = global_image.repeat(len(x["local_images"]),1).requires_grad_()
        processed={
            "local_images":[ self.process_image(i) for i in x["local_images"]],
            # "global_images":torch.vstack([ self.preprocess(x["global_image"]) for _ in x["local_images"]]).view(-1,3,224,224).to(self.device),
            "sentences":clip.tokenize(x["sentences"]).to(self.device)
        }

        feats={
            "local_images":torch.vstack([ img for img in processed["local_images"]]),
            # "global_images":self.clip.encode_image(processed["global_images"]),
            "sentences":self.clip.encode_text(processed["sentences"]).float().requires_grad_()
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
        return fusion_feat[0] @ text_feat.T
    

def build_model4(config: dict):
    model = Model4()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model