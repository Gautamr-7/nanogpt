#here we implement a decoder stack
#nn.Module is PyTorch's base class that lets a Python class behave like a neural-network component and lets PyTorch track its layers, parameters,
#  device, and training/evaluation state.
import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------
# Hyperparameters
# -----------------------------

batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100  #this is to analyze the models performance uptill now to evaluate in gaps
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 #during each evaluation average upto eval_iters batches
n_embd = 64
n_head = 4
n_layer = 4  #no of blocks
dropout = 0.0
# ------------

# -----------------------------
# Reproducibility
# -----------------------------

torch.manual_seed(1337)


# -----------------------------
# Load data
# -----------------------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# -----------------------------
# Character-level tokenizer
# -----------------------------

chars = sorted(list(set(text)))  #all characters
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}  #string to integer
itos = {i: ch for i, ch in enumerate(chars)}    

encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


# -----------------------------
# Train / validation split
# -----------------------------

data = torch.tensor(encode(text), dtype=torch.long)  #all strings in text are now numbers

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


# -----------------------------
# Get a batch
# -----------------------------

def get_batch(split):

    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size,(batch_size,)) #randomly initialized on a scale across the train/test data numbers from which data is used for X/Y

    x = torch.stack([ data[i:i + block_size]   for i in ix  ]) #stacking all respective ix's values of x together

    y = torch.stack([  data[i + 1:i + block_size + 1]   for i in ix  ])  #stacking all respective ix's values of x together--> x values shifted 1 position right

    x, y = x.to(device), y.to(device)  #transfers acc to device if cuda available that

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  #model in evaluation mode for some layers to behave differently like dropout to not randomly drop weights and batchnorm to calculate using stored running stats
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  #initialized as 0
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()  #for each batch iteration calculate all losses per iterations in a batch
        out[split] = losses.mean()   #take average loss of all stores losses
    model.train()   #set model back in training mode after evaluation
    return out

# -----------------------------
# define Head
# -----------------------------

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()

        self.head_size=head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)   #WHAT INFO DO I CONTAIN
        self.query = nn.Linear(n_embd, head_size, bias=False)   #WHAT DO I WANT
        self.value = nn.Linear(n_embd, head_size, bias=False)   #HOW MUCH OF VALUE TO TAKE FROM EACH

        self.register_buffer(                 #register_buffer-->Store this tensor as part of the module, but don't treat it as a trainable parameter.
            'tril',                           #tril is triangular lower
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape  #batch,token,n_emb
                                #The maximum is: T ≤ block_size
        k = self.key(x)  
        q = self.query(x)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size **-0.5 #acc to paper we also divide by root of head_size dk   ,also transpose of k's t and c ie -2 -1

        # causal mask: don't allow tokens to look into the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)   #softmax across last dimension ie C(emb)
        wei = self.dropout(wei)
        # weighted aggregation of values
        v = self.value(x)  # (B,T,head_size)

        out = wei @ v   # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
# -----------------------------
# Multihead Attention ie parallel processing single head attentions
# -----------------------------
#In the multi-head attention:

#Each Head produces an output of shape (B, T, head_size).
#You concatenate the num_heads heads along the last dimension:Pythonout = torch.cat([h(x) for h in self.heads], dim=-1)Because num_heads * head_size == n_embd, this gives you a tensor of shape (B, T, n_embd).
#self.proj = nn.Linear(n_embd, n_embd) is then applied to this concatenated result.
#Simply concatenating them is not enough — the model needs a learned linear transformation that can mix the information coming from all the heads.
#This is exactly analogous to the final linear layer after the attention scores in the original Transformer paper (the W^O matrix).



class MultiheadAttention(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  #Every head receives ALL 64 numbers
        self.proj=nn.Linear(n_embd,n_embd)  #the final layer after concating all individual heads
        self.dropout=nn.Dropout(dropout)  #adding dropout

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)   #concating across last dim ie features
        out=self.dropout(self.proj(out))
        return out
    
# -----------------------------
# Feed ForwarD Netwrok
# -----------------------------
#his consists of two linear transformations with a ReLU activation in between

class FeedForward(nn.Module):
     """ a simple linear layer followed by a non-linearity """

     def __init__(self,n_embd):
         super().__init__()
         self.net=nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  #The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality df f = 2048.
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
         )

     def forward(self,x):
        return self.net(x)


# -----------------------------
# Block
# -----------------------------

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self,n_embd,n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like

        super().__init__()

        head_size=n_embd // n_head   #64/4=16
        self.sa = MultiheadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,x):  #RESIDUAL 
        x = x + self.sa(self.ln1(x))   #prelayer norm
        x = x + self.ffwd(self.ln2(x))
        return x

# -----------------------------
# Bigram Language Model
# -----------------------------

class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Converts each token ID → n_embd-dimensional representation
        # e.g. vocab_size=65, n_embd=32 → each character becomes a 32-D vector

        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Gives each position (0,1,2,...,block_size-1) its own n_embd vector
        # Token embedding = WHAT the token is; positional embedding = WHERE the token is

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        self.lm_head = nn.Linear(n_embd, vocab_size)  #convert emb to vocab size ie next char
        
        # takes token representations and mixes information from relevant previous tokens
            

    def forward(self, idx, targets=None):
        B,T=idx.shape
         # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)
        # (B,T) token IDs → (B,T,n_embd) token embeddings
        # tells the model WHAT each token is

        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # (T,) positions → (T,n_embd) positional embeddings
        # tells the model WHERE each token is

        x = tok_emb + pos_emb
        # combine WHAT the token is + WHERE it is
        # (B,T,C) + (T,C) → (B,T,C) through broadcasting

        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)

        logits = self.lm_head(x)
        # contextual representation → vocab_size logits
        # e.g. (B,T,32) → (B,T,65)
        # these 65 numbers represent the model's scores for the next token

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Crop the context to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get predictions
            logits, loss = self(idx_cond)

            # Focus only on the last token
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------
# Create model
# -----------------------------

model = BigramLanguageModel(vocab_size)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# -----------------------------
# Optimizer  -- Adam
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

# -----------------------------
# Training
# -----------------------------

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of training data
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)

    # clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # backpropagation
    loss.backward()

    # update parameters
    optimizer.step()


# -----------------------------
# Generate text
# -----------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))


#Training: batches are used to update the weights.

#Evaluation: batches are used to estimate the model's average loss without processing the entire dataset.

