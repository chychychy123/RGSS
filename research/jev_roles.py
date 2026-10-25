"""JEV semantic roles control bounded supervision; never accept image GT or API keys."""
from pathlib import Path
import json,hashlib
import torch
from torch import nn
import torch.nn.functional as F
from research.config import validate_role_bank


class RoleEvidence(nn.Module):
    def __init__(self, encoder, cfg):
        import clip
        super().__init__()
        bank=validate_role_bank(cfg)
        self.names=bank['classes']
        self.descriptor_count=len(bank['descriptors'])
        self.num_classes=len(self.names)+1
        features=[]
        with torch.no_grad():
            for start in range(0,self.descriptor_count,40):
                tokens=clip.tokenize(bank['descriptors'][start:start+40]).to(next(encoder.parameters()).device)
                features.append(F.normalize(encoder.encode_text(tokens).float(),dim=-1))
        self.register_buffer('features',torch.cat(features),persistent=False)
        shared=torch.tensor(bank['shared_weights'],device=self.features.device,dtype=torch.float32)
        unique=torch.tensor(bank['unique_weights'],device=self.features.device,dtype=torch.float32)
        assert shared.shape==unique.shape==(self.num_classes**2,self.descriptor_count)
        assert torch.isfinite(shared).all() and torch.isfinite(unique).all()
        self.register_buffer('shared',shared,persistent=False)
        self.register_buffer('unique',unique,persistent=False)

    @torch.no_grad()
    def forward(self, image_features, shape):
        cosine=image_features.float()@self.features.T
        cls=(cosine[:,:1]*2).softmax(-1)
        response=cosine*(cls/cls.mean(-1,keepdim=True))
        response=response-response.mean(-1,keepdim=True)
        lo=response.amin(1,keepdim=True);hi=response.amax(1,keepdim=True)
        response=(response-lo)/(hi-lo).clamp_min(1e-6)
        return response[:,1:].transpose(1,2).reshape(image_features.shape[0],self.descriptor_count,*shape)


def role_partial_loss(logits,pseudo,prototype,labels,cues,refined_cams,bank,cfg):
    """Keep original semantic label; JEV changes singleton/pair admission and auxiliary scope."""
    num_classes=logits.shape[1]
    with torch.no_grad():
        proto=F.interpolate(prototype.detach(),pseudo.shape[-2:],mode='bilinear',align_corners=False)
        confidence,other=proto.max(1)
        present=torch.cat((torch.ones_like(labels[:,:1]),labels),1).bool()
        present_other=present.gather(1,other.flatten(1)).reshape_as(other)
        valid=pseudo!=255
        safe=pseudo.clamp(0,num_classes-1)
        disagree=(other!=pseudo)&present_other&valid
        baseline=(confidence>=.7)&disagree
        # Compute role lookup on the native CLIP/DINO grid, then align it to pixels.
        shape=cues.shape[-2:]
        primary_small=F.interpolate(safe[:,None].float(),shape,mode='nearest')[:,0].long()
        other_small=prototype.detach().argmax(1)
        pair_index=(num_classes*primary_small+other_small).flatten(1)
        visual=cues.detach().flatten(2).transpose(1,2)
        shared=(visual*bank.shared[pair_index]).sum(-1).reshape(-1,*shape)
        unique=(visual*bank.unique[pair_index]).sum(-1).reshape(-1,*shape)
        shared=F.interpolate(shared[:,None],pseudo.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        unique=F.interpolate(unique[:,None],pseudo.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        neighborhood=F.avg_pool2d(F.one_hot(other_small,num_classes).permute(0,3,1,2).float(),3,stride=1,padding=1,count_include_pad=False)
        coherence=neighborhood.gather(1,other_small[:,None])
        coherence=F.interpolate(coherence,pseudo.shape[-2:],mode='nearest')[:,0]
        # Avoid applying a coarse pair's semantic judgment to a different full-resolution pair.
        same_primary=F.interpolate(primary_small[:,None].float(),pseudo.shape[-2:],mode='nearest')[:,0].long()==safe
        same_other=F.interpolate(other_small[:,None].float(),pseudo.shape[-2:],mode='nearest')[:,0].long()==other
        aligned=same_primary&same_other
        expanded=(shared>=cfg['jev_shared_response'])&(confidence>=cfg['jev_proto_min'])&disagree&aligned
        expanded &= coherence>=cfg['jev_coherence_min']
        semantic=F.interpolate(refined_cams.detach(),pseudo.shape[-2:],mode='bilinear',align_corners=False)
        semantic=torch.cat((torch.zeros_like(semantic[:,:1]),semantic),1).gather(1,safe[:,None])[:,0]
        anchor=(unique>=cfg['jev_unique_response'])&((unique-shared)>=.15)&(semantic>=.75)&(safe>0)&valid&aligned
        # A maximum of 15% of valid pixels per image can acquire new/relaxed supervision.
        for b in range(len(expanded)):
            cap=int(valid[b].sum().item()*cfg['jev_max_fraction'])
            if int(expanded[b].sum())>cap:
                candidate=shared[b].masked_fill(~expanded[b],-1).flatten()
                keep=torch.zeros_like(candidate,dtype=torch.bool)
                if cap:keep[candidate.topk(cap).indices]=True
                expanded[b]=keep.reshape_as(expanded[b])
        selected=(baseline|expanded)&~anchor
        auxiliary_ignore=expanded&~anchor
    lp=logits.log_softmax(1)
    primary=lp.gather(1,safe[:,None]).squeeze(1)
    alternate=lp.gather(1,other[:,None]).squeeze(1)
    losses=-torch.where(selected,torch.logaddexp(primary,alternate),primary)
    bg=valid&(pseudo==0);fg=valid&(pseudo!=0)
    value=.5*((losses*bg).sum()/bg.sum().clamp_min(1)+(losses*fg).sum()/fg.sum().clamp_min(1))
    aux=pseudo.clone();aux[auxiliary_ignore]=255
    stats={'jev_set_fraction':float(auxiliary_ignore.float().mean()),
           'jev_added_pair_fraction':float((expanded&~baseline&~anchor).float().mean()),
           'jev_hard_anchor_fraction':float((anchor&baseline).float().mean()),
           'partial_fraction':float(selected.float().mean()),
           'jev_auxiliary_ignore_fraction':float(auxiliary_ignore.float().mean())}
    return value,aux,stats
