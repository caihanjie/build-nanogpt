import os

import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 6000
eval_iters = 200
eval_interval = 300
learning_rate = 3e-4
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
model_save_path = "bigram_v3.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device:",device)

torch.manual_seed(1337)

#wget https://github.com/Dreamacro/clash/releases/download/v1.12.0/clash-v1.12.0-linux-amd64.gz https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt','r',encoding='utf-8') as f:
  text = f.read()

print('length of the data',len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('all the unique characters',''.join(chars))
print('vocab size',vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)
    

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        Q = self.query(x) #(B,T,head_size)
        K = self.key(x)   #(B,T,head_size)
        V = self.value(x) #(B,T,head_size)

        wei = Q @ K.transpose(-2,-1) * C**-0.5 #(B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0,float('-inf')) #(B,T,T)
        wei = F.softmax(wei,dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        out = wei @ V #(B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.positional_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_embd + pos_embd # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:] #(B, C)
            probs = F.softmax(logits,dim =-1) # (B, C)
            idx_next = torch.multinomial(probs,num_samples=1) #(B, 1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

save_dir = os.path.dirname(model_save_path)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "iter": iter + 1,
    "config": {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "dropout": dropout,
    },
    "chars": chars,
}, model_save_path)
print(f"Model checkpoint saved to {model_save_path}")

context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
