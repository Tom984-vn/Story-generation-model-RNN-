import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------------------------------
# 1. Basic setup and configuration
# --------------------------------------------------

batch_size = 32    
block_size = 8       
max_iters = 3000      
eval_interval = 300   
learning_rate = 1e-2  #try 5e-3 or 1e-3 if too high
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
eval_iters = 200      

torch.manual_seed(1337)  #random seed

# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------

#input.txt take from og repo
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Collect all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping from character ↔ integer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Helper functions for encoding/decoding between text and numbers
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# train/valid : 0.9/0.1
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --------------------------------------------------
# 3. Create data batches
# --------------------------------------------------

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --------------------------------------------------
# 4. Evaluate model performance (without training)
# --------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # set to evaluation mode (no dropout, no gradient updates)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # back to training mode
    return out

# --------------------------------------------------
# 5. Define the Bigram model
# --------------------------------------------------

class BigramLanguageModel(nn.Module):
    """A simple model that predicts the next character directly from the current one."""

    def __init__(self, vocab_size):
        super().__init__()
        # Lookup table: for each character (index), store a vector of next-char logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # shape: (B, T, C)

        if targets is None:
            loss = None
        else:
            # Flatten for cross-entropy
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generates new characters given a starting context."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]             # only look at last time step
            probs = F.softmax(logits, dim=-1)     # turn logits into probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample next char
            idx = torch.cat((idx, idx_next), dim=1)  # append to sequence
        return idx

# --------------------------------------------------
# 6. Initialize model and optimizer
# --------------------------------------------------

model = BigramLanguageModel(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --------------------------------------------------
# 7. Training loop
# --------------------------------------------------

for iter in range(max_iters):

    # Every few steps, check the loss on train and validation sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

    # Get a batch of training data
    xb, yb = get_batch('train')

    # Forward pass and loss computation
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------------------------------------
# 8. Generate new text
# --------------------------------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print("\nGenerated text:\n")
print(decode(generated[0].tolist()))
