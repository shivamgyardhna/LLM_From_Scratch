import torch
import torch.nn as nn
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.epsilon=1e-5
    self.shift=nn.Parameter(torch.zeros(emb_dim))
    self.scale=nn.Parameter(torch.ones(emb_dim))
  def forward(self,inp):
    mean = inp.mean(dim=-1, keepdim=True) 
    var = inp.var(dim=-1, keepdim=True)
    input_Norm=(inp-mean)/torch.sqrt(var+self.epsilon)
    return input_Norm*self.scale+self.shift


class MultiHeadAttention(nn.Module):
  def __init__(self,d_in,d_out,context_length,num_heads,dropout,qkv_bias=False):
    super().__init__()
    
    self.d_in=d_in
    self.d_out=d_out
    self.num_head=num_heads
    if(num_heads==0):
      num_heads+=0.000000001
    self.head_dim=d_out//num_heads
    
    self.w_query=nn.Linear(d_in, d_out, bias=qkv_bias) #layer not the actual wt
    self.w_key=nn.Linear(d_in,d_out,bias=qkv_bias)
    self.w_value=nn.Linear(d_in, d_out,bias=qkv_bias)
    self.out_proj=nn.Linear(d_out,d_out)
    self.context_length=context_length
    self.dropout=nn.Dropout(dropout)
    
    self.qkv_bias=qkv_bias
    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))#register_buffer makes PyTorch officially aware of this tensor:here var is mask

  def forward(self,x):
    batch_size,num_token,d_in=x.shape
    
    keys=self.w_key(x)
    querys=self.w_query(x)
    values=self.w_value(x)
    
    #split using view    -- converting (batch_size,num_token,d_in) =-> (batch_size,num_token,num_head,head_dim)
    keys=keys.view(batch_size,num_token,self.num_head,self.head_dim)
    querys=querys.view(batch_size,num_token,self.num_head,self.head_dim)
    values=values.view(batch_size,num_token,self.num_head,self.head_dim)
    
    #taking the transpose and grouping according to the head  converting---> (batch_size,num_token,num_head,head_dim)=-> (batch_size,num_head,num_token,head_dim)
    keys=keys.transpose(1,2)
    querys=querys.transpose(1,2)
    values=values.transpose(1,2)
    
    atten_scores=querys@keys.transpose(2,3)
    
    mask_bool=self.mask.bool()[:num_token,:num_token]
    atten_scores.masked_fill(mask_bool,-torch.inf)
    
    attn_weigth=torch.softmax(atten_scores/keys.shape[-1]**0.5,dim=-1)
    attn_weigth=self.dropout(attn_weigth)
    
    context_vec=(attn_weigth@values).transpose(1,2)
    
    context_vec=context_vec.contiguous().view(batch_size,num_token,self.d_out)
    context_vec = self.out_proj(context_vec)
    
    return context_vec    
 
 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))
   
   
class FeedForward(nn.Module):
  def __init__(self,GPT_2_config):
    super().__init__()
    self.layer=nn.Sequential(
      nn.Linear(GPT_2_config["emb_dim"],4 * GPT_2_config["emb_dim"]), #expansion
      GELU()  ,               #activation
      nn.Linear(4 * GPT_2_config["emb_dim"], GPT_2_config["emb_dim"]) #contraction
    )
  def forward(self,x):
    return self.layer(x)  
  
cfg=GPT_2_config={
  "vocab":50257,
  "context_len":1024,
  "emb_dim":768,
  "n_head":12,
  "n_layer":12,
  "dropout":0.1,
  "qkv_bias":False  
}



class Transfomers(nn.Module):
  def __init__(self,cfg):
    super().__init__()
    self.layernorm=LayerNorm(cfg["emb_dim"])
    self.mutihead_atten=MultiHeadAttention(
      d_in=cfg["emb_dim"],
      d_out=cfg["emb_dim"],
      context_length=cfg["context_len"],
      num_heads=cfg["n_head"],
      dropout=cfg["dropout"],
      qkv_bias=False)
    self.dropout=nn.Dropout(cfg["dropout"])
    self.feed_forward=FeedForward(cfg)
    
  def forward(self,x):
    shortcut=x
    
    x=self.layernorm.forward(x)
    x=self.mutihead_atten.forward(x)
    x=self.dropout(x)
    x=x+shortcut
    
    shortcut=x
    
    x=self.layernorm(x)
    x=self.feed_forward(x)
    x=self.dropout(x)
    x=x+shortcut
    
    return x 
  
  
  
  
class GPT(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    
    self.token_emb=nn.Embedding(cfg["vocab"],cfg["emb_dim"])
    self.pos_emb=nn.Embedding(cfg["context_len"],cfg["emb_dim"])
    self.dropout=nn.Dropout(cfg["dropout"])
    self.tranformer_block= nn.Sequential(
            *[Transfomers(cfg) for _ in range(cfg["n_layer"])]
        )
    self.final_norm=LayerNorm(cfg["emb_dim"])
    self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab"])
    
    
  def forward(self,ip_batch):
      batch,seq_len=ip_batch.shape
      
      token_embd=self.token_emb(ip_batch)
      pos_embd = self.pos_emb(torch.arange(seq_len, device=ip_batch.device))
      
      x=token_embd+pos_embd
      
      x=self.dropout(x)
      
      x=self.tranformer_block(x)
      
      x=self.final_norm(x)
      
      logit=self.out_head(x)
      
      return logit
       
       
       
def generate_text(model,ip_token_id,max_new_tokens,context_size):
  for i in range(max_new_tokens):
    context_ip=ip_token_id[:,-context_size:]
    with torch.no_grad():
      logit=model(context_ip)
    logit=logit[:,-1,:] #tensor size is (batch) x 1 x (vocab_size)  
    prob=torch.softmax(logit,dim=-1)#convert to prob , we can take max of logit also that give same but prob give explainibilty
    idx_next=torch.argmax(prob,dim=-1,keepdim=True) #taking max val prob
    ip_token_id=torch.concat((ip_token_id,idx_next),dim=1)
  return ip_token_id  


#Encoding text --> token_id
def text_to_token_id(text,tokenizer=tokenizer):
  text_batch=[]
  for i in range(len(text)):
    text_batch.append(torch.tensor(tokenizer.encode(text[i])))
  text_batch=torch.stack(text_batch)
  return text_batch  

#decoding token_id -->text
def token_id_to_text(token_id,tokenizer=tokenizer):
  text_batch=[]
  for i in range(len(token_id)):
    text=tokenizer.decode(token_id[i].squeeze(0).tolist())
    text_batch.append(text)
  return  text_batch                    