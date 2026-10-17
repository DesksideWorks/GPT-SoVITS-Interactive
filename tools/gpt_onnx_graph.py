"""Fixed-shape GPT graphs for the deployed DirectML semantic backend."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Graph(nn.Module):
    def __init__(self,core,capacity):
        super().__init__();self.core=core;self.capacity=capacity;self.heads=core.num_head;self.dim=core.model_dim;self.hd=self.dim//self.heads
        assert not core.norm_first
    def layer(self,x,layer,k=None,v=None,position=None,bias=None):
        q,new_k,new_v=F.linear(x,layer.self_attn.in_proj_weight,layer.self_attn.in_proj_bias).chunk(3,dim=-1)
        if k is None:k,v=new_k,new_v
        else:
            index=position.reshape(1,1,1).expand(1,1,self.dim)
            k=torch.scatter(k,1,index,new_k);v=torch.scatter(v,1,index,new_v)
        qh=q.reshape(1,-1,self.heads,self.hd).transpose(1,2)
        kh=k.reshape(1,-1,self.heads,self.hd).transpose(1,2);vh=v.reshape(1,-1,self.heads,self.hd).transpose(1,2)
        scores=torch.matmul(qh,kh.transpose(-2,-1))*(1/math.sqrt(self.hd))
        if bias is not None:scores=scores+bias
        attention=torch.matmul(torch.softmax(scores,-1),vh).transpose(1,2).reshape(1,-1,self.dim)
        x=F.layer_norm(x+F.linear(attention,layer.self_attn.out_proj.weight,layer.self_attn.out_proj.bias),[self.dim],layer.norm1.weight,layer.norm1.bias,layer.norm1.eps)
        ff=F.linear(F.relu(F.linear(x,layer.linear1.weight,layer.linear1.bias)),layer.linear2.weight,layer.linear2.bias)
        x=F.layer_norm(x+ff,[self.dim],layer.norm2.weight,layer.norm2.bias,layer.norm2.eps)
        return x,k,v

class Prefill(Graph):
    def forward(self,phones,bert):
        embedded=self.core.ar_text_embedding(phones)+self.core.bert_proj(bert.transpose(1,2))
        x=self.core.ar_text_position(embedded);keys=[];values=[]
        for layer in self.core.h.layers:
            x,k,v=self.layer(x,layer)
            keys.append(F.pad(k,(0,0,0,self.capacity-k.shape[1])))
            values.append(F.pad(v,(0,0,0,self.capacity-v.shape[1])))
        return self.core.ar_predict_layer(x[:,-1]),torch.stack(keys),torch.stack(values)

class Step(Graph):
    def forward(self,token,position,audio_position,attention_bias,cache_k,cache_v):
        embedded=self.core.ar_audio_embedding(token)
        pe=torch.index_select(self.core.ar_audio_position.pe[0],0,audio_position).unsqueeze(0)
        x=embedded*self.core.ar_audio_position.x_scale+self.core.ar_audio_position.alpha*pe
        keys=[];values=[]
        for i,layer in enumerate(self.core.h.layers):
            x,k,v=self.layer(x,layer,cache_k[i],cache_v[i],position,attention_bias)
            keys.append(k);values.append(v)
        return self.core.ar_predict_layer(x[:,0]),torch.stack(keys),torch.stack(values)

class PrefillBucket(Graph):
    def forward(self,phones,bert,attention_bias,last_position):
        embedded=self.core.ar_text_embedding(phones)+self.core.bert_proj(bert.transpose(1,2))
        x=self.core.ar_text_position(embedded);keys=[];values=[]
        for layer in self.core.h.layers:
            x,k,v=self.layer(x,layer,bias=attention_bias)
            keys.append(F.pad(k,(0,0,0,self.capacity-k.shape[1])))
            values.append(F.pad(v,(0,0,0,self.capacity-v.shape[1])))
        last=torch.index_select(x,1,last_position)[:,0]
        return self.core.ar_predict_layer(last),torch.stack(keys),torch.stack(values)


