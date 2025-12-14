import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size = 4
block_size = 8
max_iters = 5000
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2
n_embd = 32


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
        self.met = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.met(x)

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        Q = self.query(x) #(B,T,head_size)
        K = self.key(x)   #(B,T,head_size)
        V = self.value(x) #(B,T,head_size)

        wei = Q @ K.transpose(-2,-1) * C**-0.5 #(B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0,float('-inf')) #(B,T,T)
        wei = F.softmax(wei,dim=-1) #(B,T,T)
        out = wei @ V #(B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)
        return out
    

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(n_head=4, head_size=n_embd//4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.positional_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_embd + pos_embd # (B,T,C)
        x = self.sa_heads(x) # (B,T,C)
        x = self.ffwd(x) # (B,T,C)
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

context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
