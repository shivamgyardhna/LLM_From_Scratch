#importing Library and function
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
from GPT_Model import GPT,generate_text,cfg,tokenizer

#text data
with open("the-verdict.txt",'r') as file:
  text=file.read()
total_characters = len(text)
total_tokens = len(tokenizer.encode(text))  


import tiktoken
from torch.utils.data import Dataset, DataLoader

# create (input,target)
class GPTdataset(Dataset):  
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids =[]
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

##create dataloader with random sampling using dataloader
def createdataloader(txt,batch_size,max_len,stride,shuffle=True,drop_last=True,num_worker=0):  
  tokenizer=tiktoken.get_encoding("gpt2")
  dataset=GPTdataset(txt,tokenizer,max_len,stride)
  dataloader=DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_worker
  )
  return dataloader

#dividing data 
train_ratio=.8
n=len(text)
train_data=text[:int(n*train_ratio)]
val_data=text[int(n*train_ratio)+1:]

torch.manual_seed(123) #to make the consistent random value

#dataloader for train
train_loader = createdataloader( 
    txt=train_data,
    batch_size=2,
    max_len=cfg["context_len"],
    stride=cfg["context_len"],
    drop_last=True,
    shuffle=True,
    num_worker=0
)
#dataloader for val
val_loader=createdataloader(
     txt=val_data,
    batch_size=2,
    max_len=cfg["context_len"],
    stride=cfg["context_len"],
    drop_last=True,
    shuffle=True,
    num_worker=0
)

# CHECK IF TOTAL TOKENS IS NOT LESS THAN CONTEXT SIZE
if total_tokens*train_ratio < cfg["context_len"]/4:
    print("NOT ENOUGH TOKENS FOR THE TRAINING LOADER")

if total_tokens*(1-train_ratio) < cfg["context_len"]/4:
    print("NOT ENOUGH TOKENS FOR THE VALIDATION LOADER")

#cal loss for one batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

#cal loss for whole data loaders
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss+=loss.item()

        else:
            break

    return total_loss/num_batches

def text_to_token_id(text, tokenizer):
    encoded = tokenizer.encode(text)
    # encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #Add batch dimension
    return encoded_tensor

def token_id_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) #remove batch dimension
    return tokenizer.decode(flat.tolist())

#cal train ,val loss
def evaluate_model(model,train_loader,val_loader,device,eval_iter):
  model.eval()
  with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device,num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device,num_batches=eval_iter)

    model.train()
    return train_loss,val_loss
  

#printing text for given model 50 next token
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = cfg["context_len"]
    encoded = text_to_token_id([start_context], tokenizer).to(device)  # wrap in list

    with torch.no_grad():
        token_ids = generate_text(
            model=model, ip_token_id=encoded, max_new_tokens=50, context_size=context_size
        )

    decoded_text = token_id_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # index into list
    model.train()



#train the model and it return train,val,token seen
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:  # ← inner loop
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch+1} (step: {global_step}): Train_Loss={train_loss}, Val_Loss={val_loss}")


        print("\n####################################\n")
        print(f"Epoch {epoch+1} sample output")
        generate_and_print_sample(model, tokenizer, device, start_context)
        print("\n####################################\n")

    return train_losses, val_losses, track_tokens_seen



#main function do model train also find time take by model then plot the graph of training and validation
if __name__=='__main__':
  import time
  start_time=time.time()

  torch.manual_seed(123)
  model=GPT(cfg)
  model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
  # model.to(device=torch.device("cpu"))

  optimizer=torch.optim.AdamW(model.parameters(),lr=0.005,weight_decay=0.1)

  epochs=100
  train_losses,val_losses,track_tokens_seen=train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=epochs, eval_freq=5, eval_iter=10, start_context="Every effort moves you", tokenizer=tokenizer)
  end_time = time.time()

  print("Execution time: ", (end_time-start_time)/60, "minutes")
  ################################################
  print("Loss-Plot")
  import matplotlib.pylab as plt
  from matplotlib.ticker import MaxNLocator

  def plotlosses(epochs,token_seen,train_losses,val_losses):
    fig,ax=plt.subplots(figsize=(5,6))

    ax.plot(epochs,train_losses,label="Trining Loss")
    ax.plot(epochs,val_losses,linestyle="--",label="val_loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax.twiny()
    ax2.plot(token_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()


  epochs_tensor = torch.linspace(0, epochs, len(train_losses))
  plotlosses(epochs_tensor, track_tokens_seen, train_losses, val_losses)