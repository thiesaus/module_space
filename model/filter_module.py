import torch
from torch import nn,randn
import clip
import numpy as np
from model.position_embedding import build as build_position_embedding
import torch.nn.functional as F
from utils.utils import distributed_rank

class FilterModule(nn.Module):
    def __init__(self, ):
        super(FilterModule, self).__init__()
        model, preprocess = clip.load("RN50")
        self.clip=model
        self.preprocess=preprocess
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._freeze_encoder()
        local_reso = 32 * 32
        local_scale = local_reso ** -0.5
        self.pos_emb_local = nn.Parameter(local_scale * randn(local_reso)).to(self.device)
        global_reso = 32 * 32
        global_scale = global_reso ** -0.5
        self.pos_emb_global = nn.Parameter(global_scale * randn(global_reso)).to(self.device)
        self.img_dim = 1024
        self.text_dim = 1024
        self.fusion_fc = nn.Linear(self.text_dim, self.img_dim).to(self.device)
        self.fusion_local_global = nn.MultiheadAttention(
            embed_dim=self.img_dim,
            num_heads=4,
            dropout=0.,
        ).to(self.device)
        self.fusion_visual_textual = nn.MultiheadAttention(
                embed_dim=self.img_dim,
                num_heads=4,
                dropout=0,
            ).to(self.device)

        self.middle_layer = nn.MultiheadAttention(
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
        batch_feats=[self.processing_input(i) for i in x]

        quantities=[batch["local_images"].shape[0] for batch in batch_feats]
        norm_feats={k:torch.vstack([i[k] for i in batch_feats]) for k in batch_feats[0].keys()}

        #fusion local_global
        local_feat=norm_feats["local_images"].requires_grad_()
        global_feat=norm_feats["global_images"].requires_grad_()
        text_feat=norm_feats["sentences"].requires_grad_()

        local_global_feat=self.local_global(local_feat, global_feat)
        text_local_feat=self.textual_local(local_feat, text_feat)
        text_global_feat=self.textual_global(global_feat, text_feat)

        gt_gl_feat=self.gt_gl_fusion(text_global_feat, local_global_feat)
        gl_lt_feat=self.gl_lt_fusion(local_global_feat, text_local_feat)

        full_feat=self.full_fusion(gt_gl_feat, gl_lt_feat)

        fusion_feat=self.repeat_text(text_feat, full_feat)

        all_logits=[]
        for i,quantity in enumerate(quantities):
            ff=fusion_feat[i*quantity:(i+1)*quantity] if i+1<len(quantities) else fusion_feat[i*quantity:]
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
        global_image = self.preprocess(global_image).unsqueeze(0).to(self.device)
        global_image = self.clip.encode_image(global_image).float()
        global_image = global_image / global_image.norm(dim=-1, keepdim=True)
        global_image = global_image.repeat(len(x["local_images"]),1)
        processed={
            "local_images":[ self.preprocess(i).to(self.device) for i in x["local_images"]],
            # "global_images":torch.vstack([ self.preprocess(x["global_image"]) for _ in x["local_images"]]).view(-1,3,224,224).to(self.device),
            "sentences":clip.tokenize(x["sentences"]).to(self.device)
        }

        feats={
            "local_images":torch.vstack([ self.clip.encode_image(img.unsqueeze(0)).float() for img in processed["local_images"]]),
            # "global_images":self.clip.encode_image(processed["global_images"]),
            "sentences":self.clip.encode_text(processed["sentences"]).float()
        }
        norm_feats={
            "local_images":feats["local_images"]/feats["local_images"].norm(dim=-1, keepdim=True),
            # "global_images":feats["global_images"]/feats["global_images"].norm(dim=-1, keepdim=True),
            "global_images":global_image,
            "sentences":feats["sentences"]/feats["sentences"].norm(dim=-1, keepdim=True)
        }
   
        return norm_feats
    
    def local_global(self, local_image, global_image):
        local_feat=local_image + self.pos_emb_local
        global_feat=global_image + self.pos_emb_global
        fusion_feat= self.fusion_local_global(
            query=local_feat,
            key=global_feat,
            value=global_feat
        )
        return fusion_feat[0] + local_feat

    def textual_local(self, local_image, text_feat):
        text_feat = self.fusion_fc(text_feat)
        local_feat=local_image + self.pos_emb_local
        fusion_feat= self.fusion_visual_textual(
            query=text_feat,
            key=local_feat,
            value=local_feat
        )
        return fusion_feat[0] + local_feat
    
    def textual_global(self, global_image, text_feat):
        text_feat = self.fusion_fc(text_feat)
        global_feat=global_image + self.pos_emb_global
        fusion_feat= self.fusion_visual_textual(
            query=text_feat,
            key=global_feat,
            value=global_feat
        )
        return fusion_feat[0] + text_feat
    
    def gt_gl_fusion(self,gt_feat,gl_feat):
        fusion_feat= self.middle_layer(
            query=gl_feat,
            key=gt_feat,
            value=gt_feat
        )
        return fusion_feat[0]

    def gl_lt_fusion(self,gl_feat,lt_feat):
        fusion_feat= self.middle_layer(
            query=lt_feat,
            key=gl_feat,
            value=gl_feat
        )
        return fusion_feat[0]
    
    def full_fusion(self,gt_gl_feat,gl_lt_feat):
        fusion_feat= self.middle_layer(
            query=gl_lt_feat,
            key=gt_gl_feat,
            value=gt_gl_feat
        )
        return fusion_feat[0]
    
    def repeat_text(self, text_feat, full_feat):
        text_feat = self.fusion_fc(text_feat)
        fusion_feat= self.middle_layer(
            query=text_feat,
            key=full_feat,
            value=full_feat
        )
        return fusion_feat[0] * text_feat
    

def build_model(config: dict):
    model = FilterModule()
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))

    return model