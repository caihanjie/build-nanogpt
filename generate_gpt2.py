import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    def __init__(self, block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.config.block_size}")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", GPTConfig())
    if isinstance(cfg, dict):
        cfg = GPTConfig(**cfg)
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, cfg


def generate(model, prompt, max_new_tokens, top_k, temperature, device, num_return_sequences=1, seed=42):
    enc = tiktoken.get_encoding("gpt2")
    start_tokens = enc.encode(prompt)
    x = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)
    x = x.repeat(num_return_sequences, 1)
    sample_rng = torch.Generator(device=device)
    if seed is not None:
        sample_rng.manual_seed(seed)
    for _ in range(max_new_tokens):
        idx_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size :]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=sample_rng)
        x = torch.cat((x, next_token), dim=1)
    return [enc.decode(seq.tolist()) for seq in x]


def pick_device(cli_device):
    if cli_device:
        return cli_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Load a saved GPT model and generate text.")
    parser.add_argument("--checkpoint", default="log/model_latest.pt", help="path to checkpoint produced by train_gpt2.py")
    parser.add_argument("--prompt", default="Hello, I'm a language model,", help="prompt to start generation")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="number of tokens to sample")
    parser.add_argument("--top-k", type=int, default=50, help="restrict sampling to top-k tokens (set 0 to disable)")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    parser.add_argument("--num-return-sequences", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    parser.add_argument("--device", default=None, help="force device (cuda, cpu, mps); auto if omitted")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"using device: {device}")
    model, cfg = load_model(args.checkpoint, device)
    print(
        f"loaded checkpoint from {args.checkpoint} | layers {cfg.n_layer}, heads {cfg.n_head}, emb {cfg.n_embd}, block {cfg.block_size}"
    )
    outputs = generate(
        model=model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k if args.top_k > 0 else None,
        temperature=args.temperature,
        device=device,
        num_return_sequences=args.num_return_sequences,
        seed=args.seed,
    )
    for i, text in enumerate(outputs):
        print(f"[sample {i}] {text}")


if __name__ == "__main__":
    main()
