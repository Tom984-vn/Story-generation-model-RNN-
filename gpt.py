import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters ---
BATCH_SIZE = 64
BLOCK_SIZE = 256
EMBED_DIM = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
LR = 3e-4
EVAL_ITERS = 200
EVAL_INTERVAL = 500
MAX_ITERS = 5000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# --- Load and encode data ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda t: ''.join([itos[i] for i in t])

data = encode(text)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    src = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([src[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([src[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Model definition ---
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_dim = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(DROPOUT)
        self.n_head = n_head
        self.scale = head_dim ** -0.5
        self.tril = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).to(DEVICE)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, -1).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, -1).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * self.scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))

class FeedForward(nn.Sequential):
    def __init__(self, n_embd):
        super().__init__(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = SelfAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ff(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[TransformerBlock(EMBED_DIM, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=DEVICE))
        x = self.blocks(tok + pos)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Training ---
model = GPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        loss_dict = estimate_loss(model)
        print(f"Step {step}: train {loss_dict['train']:.4f}, val {loss_dict['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- Text generation ---
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, 500)[0].tolist()))
