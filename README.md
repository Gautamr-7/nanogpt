# nanogpt

Absolutely. Let's turn **your exact code** into a revision sheet. The goal is that later you can look at this and reconstruct the whole GPT without memorizing lines.

One important thing first: **there is one bug in your current code**:

```python
wei = q @ k.transpose(-2, -1) * head_size ** -0.5
```

`head_size` is not defined inside `Head.forward()`. Store it in `self.head_size` or use `k.size(-1)`.

I would write:

```python
self.head_size = head_size
...
wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
```

Also, your comment:

```python
v = self.value(x)  # (B,T,C)
```

is technically wrong. It is:

```text
(B, T, head_size)
```

With your settings:

```text
n_embd = 64
n_head = 4
head_size = 64 / 4 = 16
```

So `v` is `(B,T,16)`.

---

# 1. Big picture

Your model does this:

```text
characters
   ↓
integer token IDs
   ↓
token embeddings
   +
positional embeddings
   ↓
(B,T,64)
   ↓
Transformer Block
   ├── Self-Attention
   └── Feed Forward
   ↓
repeat 4 times
   ↓
LayerNorm
   ↓
Linear
   ↓
vocab logits
   ↓
Cross Entropy Loss
```

The fundamental idea is:

> **Attention allows tokens to communicate with previous tokens. FFN processes/refines the information at each token.**

---

# 2. Hyperparameters

```python
batch_size = 16
```

We process **16 independent sequences simultaneously**.

Shape:

```text
B = 16
```

---

```python
block_size = 32
```

Maximum context length.

Each training example contains at most:

```text
32 tokens
```

So:

```text
T = 32
```

---

```python
n_embd = 64
```

Every token is represented using **64 numbers**.

Therefore:

```text
token embedding = 64-dimensional vector
```

This is your model's main representation dimension.

---

```python
n_head = 4
```

Each Transformer attention layer has 4 independent attention heads.

Because:

```text
64 embedding dimensions / 4 heads
= 16 dimensions per head
```

Therefore:

```python
head_size = 16
```

---

```python
n_layer = 4
```

You stack 4 Transformer blocks:

```text
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
Block 4
```

---

# 3. Reading the dataset

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

Loads your training text as one giant string.

For example:

```text
"hello world"
```

---

# 4. Character tokenizer

```python
chars = sorted(list(set(text)))
```

Find every unique character.

Suppose:

```text
text = "hello"
```

Then:

```text
chars = ['e', 'h', 'l', 'o']
```

---

```python
vocab_size = len(chars)
```

Number of unique tokens.

Here:

```text
vocab_size = 4
```

---

```python
stoi = {ch: i for i, ch in enumerate(chars)}
```

String → integer.

Example:

```text
'e' → 0
'h' → 1
'l' → 2
'o' → 3
```

---

```python
itos = {i: ch for i, ch in enumerate(chars)}
```

Reverse:

```text
0 → e
1 → h
2 → l
3 → o
```

---

```python
encode = lambda s: [stoi[c] for c in s]
```

Turns:

```text
"hello"
```

into:

```text
[1,0,2,2,3]
```

---

```python
decode = lambda l: ''.join([itos[i] for i in l])
```

Turns the numbers back into characters.

---

# 5. Train/validation split

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

Your entire text is now integers.

Example:

```text
hello
↓
[1,0,2,2,3]
```

---

```python
n = int(0.9 * len(data))
```

90% for training.

---

```python
train_data = data[:n]
val_data = data[n:]
```

You train on one portion and evaluate on unseen data.

---

# 6. `get_batch()`

This is extremely important.

```python
ix = torch.randint(
    len(data) - block_size,
    (batch_size,)
)
```

Randomly chooses starting positions.

Suppose:

```text
block_size = 4
```

and:

```text
data = "abcdefgh..."
```

If:

```text
i = 3
```

then:

```python
x = data[3:7]
```

gives:

```text
d e f g
```

But the target is shifted one character:

```python
y = data[4:8]
```

giving:

```text
e f g h
```

Therefore:

```text
X: d e f g
Y: e f g h
```

The task is:

```text
d → predict e
e → predict f
f → predict g
g → predict h
```

This is **next-token prediction**.

---

# 7. The initial shape

With:

```python
batch_size = 16
block_size = 32
```

you get:

```text
x.shape = (16, 32)
```

Meaning:

```text
16 sequences
32 token IDs per sequence
```

But these are still integers.

---

# 8. Token embedding

```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
```

This creates a learnable lookup table:

```text
vocab_size × 64
```

Suppose:

```text
vocab_size = 65
n_embd = 64
```

then:

```text
65 × 64
```

Every character has a learnable 64-dimensional vector.

---

```python
tok_emb = self.token_embedding_table(idx)
```

Input:

```text
(B,T)
```

Output:

```text
(B,T,64)
```

So:

```text
(16,32)
      ↓
(16,32,64)
```

This is:

> **WHAT is this token?**

---

# 9. Positional embedding

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

Every position gets its own learnable vector.

For example:

```text
position 0 → 64 numbers
position 1 → 64 numbers
position 2 → 64 numbers
...
position 31 → 64 numbers
```

---

```python
pos_emb = self.position_embedding_table(
    torch.arange(T, device=device)
)
```

Shape:

```text
(T,64)
```

or:

```text
(32,64)
```

This tells the model:

> **WHERE is this token?**

---

# 10. Combine them

```python
x = tok_emb + pos_emb
```

We have:

```text
tok_emb = (16,32,64)
pos_emb =      (32,64)
```

PyTorch broadcasts the positional embeddings across the batch:

```text
(16,32,64)
+
(   32,64)
──────────
(16,32,64)
```

Now each token representation contains:

```text
WHAT token
+
WHERE token
```

---

# 11. Transformer Block

```python
x = self.blocks(x)
```

You have:

```python
n_layer = 4
```

so:

```text
x
 ↓
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
Block 4
```

The shape remains:

```text
(B,T,64)
```

throughout.

But **the information inside those 64 numbers changes**.

That's crucial.

Same shape ≠ same representation.

---

# 12. Inside one Block

Your block:

```python
class Block(nn.Module):

    def __init__(self,n_embd,n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiheadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
```

With your settings:

```text
n_embd = 64
n_head = 4

head_size = 64 / 4
          = 16
```

---

# 13. First operation: LayerNorm

```python
self.ln1(x)
```

LayerNorm normalizes the 64-dimensional representation **for each token**.

Shape doesn't change:

```text
(B,T,64)
→
(B,T,64)
```

It helps make training more stable.

---

# 14. Self-attention

```python
self.sa(self.ln1(x))
```

This is where tokens communicate.

---

## Q/K/V

Inside `Head`:

```python
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

Your input:

```text
64 numbers
```

gets transformed into three different representations:

```text
64 → 16   Query
64 → 16   Key
64 → 16   Value
```

For every token.

So:

```text
x
 ↓
Q: (B,T,16)
K: (B,T,16)
V: (B,T,16)
```

### Conceptually

**Query:**

> "What information am I looking for?"

**Key:**

> "What kind of information do I contain?"

**Value:**

> "Here is the actual information you can take from me."

---

# 15. Attention score

```python
wei = q @ k.transpose(-2,-1)
```

Shapes:

```text
q = (B,T,16)

k = (B,T,16)

k.T = (B,16,T)
```

Therefore:

```text
(B,T,16)
@
(B,16,T)

↓

(B,T,T)
```

That final matrix is extremely important.

For every token, it tells us:

> **How much should I pay attention to every other token?**

For example, with 4 tokens:

```text
        token 0 token 1 token 2 token 3
token 0   1      0      0      0
token 1  .3     .7      0      0
token 2  .1     .6     .3      0
token 3  .2     .1     .5     .2
```

---

# 16. Why divide by √head_size?

Your code:

```python
wei = q @ k.transpose(-2,-1) * self.head_size ** -0.5
```

is equivalent to:

```python
wei = (q @ k.transpose(-2,-1)) / math.sqrt(head_size)
```

With:

```text
head_size = 16
```

we divide by:

```text
√16 = 4
```

The Transformer paper uses this scaling because dot products tend to become larger as the dimensionality increases.

If scores become extremely large, softmax can become very sharp:

```text
[0.000001, 0.999999]
```

which gives poor gradients.

Scaling keeps the values in a more useful range.

**Important:** this is `head_size`, not vocabulary size.

---

# 17. Causal masking

```python
wei = wei.masked_fill(
    self.tril[:T,:T] == 0,
    float('-inf')
)
```

GPT cannot see the future.

For:

```text
The cat sat
```

when predicting after `"The"`:

```text
The → can see The
```

but cannot see:

```text
cat
sat
```

The mask looks like:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

`0` positions become:

```text
-inf
```

Then softmax turns those into:

```text
0
```

So future tokens receive zero attention.

---

# 18. Softmax

```python
wei = F.softmax(wei, dim=-1)
```

Converts raw scores into attention probabilities.

Example:

```text
scores:
[2.0, 1.0, 0.0]
```

becomes approximately:

```text
[0.665, 0.245, 0.090]
```

They sum to:

```text
1.0
```

Now we have actual attention weights.

---

# 19. Dropout

```python
wei = self.dropout(wei)
```

During training, some attention weights are randomly dropped.

This acts as regularization.

During evaluation/generation, dropout is disabled.

---

# 20. Value

```python
v = self.value(x)
```

This is the part you struggled with earlier.

Suppose:

```text
attention weights:
[0.1, 0.7, 0.2]
```

and values:

```text
V0
V1
V2
```

Then:

```text
output =
0.1V0 + 0.7V1 + 0.2V2
```

So attention is essentially:

> **Use Q/K to decide where to look, then use those weights to mix the Vectors.**

That's why Q/K alone isn't enough.

Q/K determines **WHERE to look**.

V determines **WHAT information to retrieve**.

---

# 21. One head output

For your model:

```text
wei = (B,T,T)
v   = (B,T,16)
```

Therefore:

```python
out = wei @ v
```

gives:

```text
(B,T,T)
@
(B,T,16)

↓

(B,T,16)
```

So **one head produces 16 features per token**.

---

# 22. Multi-head attention

You have:

```python
self.heads = nn.ModuleList(
    [Head(head_size) for _ in range(num_heads)]
)
```

You create 4 separate Heads.

Each head receives the **entire 64-dimensional input**.

This is important:

```text
                 64 dimensions
                     │
          ┌──────────┼──────────┐
          ↓          ↓          ↓
        Head 1     Head 2     Head 3 ... Head 4
         16          16         16       16
```

They don't receive:

```text
Head 1 → dimensions 0-15
Head 2 → dimensions 16-31
...
```

Instead, **each head has its own learned Q/K/V projections**:

```text
64 → 16
```

So each head can learn a different way of looking at the same representation.

---

# 23. Concatenate heads

```python
out = torch.cat(
    [h(x) for h in self.heads],
    dim=-1
)
```

Each head:

```text
(B,T,16)
```

Four heads:

```text
(B,T,16)
(B,T,16)
(B,T,16)
(B,T,16)
```

Concatenate:

```text
(B,T,64)
```

because:

```text
16 + 16 + 16 + 16 = 64
```

---

# 24. Projection

Then:

```python
self.proj = nn.Linear(n_embd,n_embd)
```

and:

```python
out = self.proj(out)
```

So:

```text
64 → 64
```

Why?

The four heads independently produced information:

```text
Head 1 ──┐
Head 2 ──┤
Head 3 ──┼── concatenate → 64
Head 4 ──┘
```

The projection is a learned transformation that **mixes information across those concatenated head outputs**.

It's not changing the size.

It's changing the representation.

---

# 25. Residual connection

Your Block:

```python
x = x + self.sa(self.ln1(x))
```

This is:

```text
original x
    +
attention(x)
    ↓
new x
```

Why?

Because the model can preserve the original representation while adding the newly learned information.

It also provides a much easier path for gradients to flow backward.

---

# 26. Feed Forward Network

Then:

```python
x = x + self.ffwd(self.ln2(x))
```

Your FFN:

```python
nn.Linear(n_embd, 4 * n_embd)
nn.ReLU()
nn.Linear(4 * n_embd, n_embd)
```

With `n_embd = 64`:

```text
64
 ↓
256
 ↓
ReLU
 ↓
64
```

Notice:

```text
input  = 64
output = 64
```

but internally:

```text
64 → 256 → 64
```

Why expand?

The larger intermediate space gives the network more capacity to perform nonlinear transformations.

And remember the division of labor:

```text
Attention:
"Which tokens should communicate?"

FFN:
"Now that I've received this information,
what computation/transformation should I perform?"
```

---

# 27. Second residual

```python
x = x + self.ffwd(self.ln2(x))
```

Again:

```text
x
+
FFN(x)
```

So the whole block is essentially:

```text
                    ┌──────────────┐
                    ↓              │
x → LayerNorm → Attention → + ─────┘
                              │
                              ↓
                         LayerNorm
                              ↓
                            FFN
                              ↓
                            + x
                              ↓
                           output
```

---

# 28. Repeat blocks

```python
self.blocks = nn.Sequential(
    *[Block(...) for _ in range(n_layer)]
)
```

With:

```python
n_layer = 4
```

you get:

```text
Block 1
   ↓
Block 2
   ↓
Block 3
   ↓
Block 4
```

The representation stays:

```text
(B,T,64)
```

but becomes increasingly contextualized.

Think:

```text
Embedding:
"What is this token?"

After Block 1:
"What nearby information matters?"

After Block 2:
"What relationships are emerging?"

After Block 3:
"What higher-level pattern is present?"

After Block 4:
"What information is useful for predicting the next token?"
```

That's conceptual, not a strict division of functionality between layers.

---

# 29. Final LayerNorm

```python
x = self.ln_f(x)
```

Still:

```text
(B,T,64)
```

This normalizes the final representations before prediction.

---

# 30. LM head

```python
self.lm_head = nn.Linear(n_embd, vocab_size)
```

This is where the 64-dimensional representation becomes scores for **every possible next character**.

Suppose:

```text
n_embd = 64
vocab_size = 65
```

Then:

```text
64 → 65
```

For every token position.

So:

```text
(B,T,64)
     ↓
(B,T,65)
```

Each set of 65 numbers means:

```text
score for 'a'
score for 'b'
score for 'c'
...
score for 'z'
...
```

These are called **logits**.

---

# 31. Why `lm_head`?

This is the final translator:

```text
internal representation
        ↓
     lm_head
        ↓
possible vocabulary
```

For example:

```text
representation of "The ca"
             ↓
       lm_head
             ↓
'a' → 2.1
'b' → -1.2
't' → 0.7
...
```

Then softmax converts these into probabilities.

---

# 32. Cross entropy

During training:

```python
loss = F.cross_entropy(logits, targets)
```

The model predicted:

```text
"a": 0.05
"t": 0.80
"r": 0.03
...
```

but the correct next character was:

```text
"a"
```

The model gets penalized because it gave `a` low probability.

Training tries to adjust the parameters so that:

```text
P(correct next token)
```

gets larger.

---

# 33. Why reshape logits?

Originally:

```text
logits = (B,T,vocab_size)
```

Example:

```text
(16,32,65)
```

But `cross_entropy` expects the classification dimension in a convenient 2D form:

```text
(B*T, vocab_size)
```

So:

```python
logits = logits.view(B*T, C)
```

becomes:

```text
(512,65)
```

because:

```text
16 × 32 = 512
```

Targets:

```text
(16,32)
```

become:

```text
(512)
```

Now the loss sees:

```text
512 predictions
each with 65 possible classes
```

---

# 34. Training loop

The core training cycle is:

```python
xb, yb = get_batch('train')
```

Get examples.

↓

```python
logits, loss = model(xb, yb)
```

Forward pass.

↓

```python
optimizer.zero_grad()
```

Clear old gradients.

↓

```python
loss.backward()
```

Backpropagation.

Compute:

```text
∂loss / ∂parameter
```

for all learnable parameters.

↓

```python
optimizer.step()
```

Update parameters.

So:

```text
data
 ↓
forward
 ↓
prediction
 ↓
loss
 ↓
backprop
 ↓
gradients
 ↓
AdamW
 ↓
new weights
```

Repeat thousands of times.

---

# 35. `estimate_loss()`

```python
@torch.no_grad()
```

means:

> Don't calculate/store gradients here.

Because we're only evaluating.

```python
model.eval()
```

puts model into evaluation mode.

This matters especially for dropout.

Then:

```python
losses[k] = loss.item()
```

collects many losses.

Finally:

```python
losses.mean()
```

gives a more reliable estimate of average loss.

---

# 36. Generation

This is another section you should understand very well.

You start:

```python
context = torch.zeros((1,1), dtype=torch.long)
```

One sequence containing one token.

Then:

```python
for _ in range(max_new_tokens):
```

Generate one token at a time.

---

### Step 1

```python
logits, loss = self(idx_cond)
```

Feed the current context into GPT.

Suppose:

```text
"The ca"
```

---

### Step 2

```python
logits = logits[:, -1, :]
```

We only care about the **last token's prediction**.

Shape:

```text
(B,T,vocab)
```

becomes:

```text
(B,vocab)
```

---

### Step 3

```python
probs = F.softmax(logits, dim=-1)
```

Turn logits into probabilities.

---

### Step 4

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

Sample a token.

Suppose model says:

```text
'a' = 0.7
't' = 0.2
'r' = 0.1
```

It may sample:

```text
'a'
```

---

### Step 5

```python
idx = torch.cat((idx, idx_next), dim=1)
```

Append the new token.

Now:

```text
"The ca"
```

becomes:

```text
"The cat"
```

Then feed it back into the model.

This repeats.

---

# 37. The entire model in one diagram

Memorize this, not the code:

```text
               TOKEN IDs
                   │
                   ↓
          Token Embedding
                   │
                   +
          Positional Embedding
                   │
                   ↓
             (B,T,64)
                   │
          ┌────────▼────────┐
          │ Transformer     │
          │ Block           │
          │                 │
          │ LayerNorm       │
          │      ↓          │
          │ MultiHead       │
          │ Attention       │
          │      ↓          │
          │   Residual      │
          │      ↓          │
          │ LayerNorm       │
          │      ↓          │
          │ FeedForward     │
          │      ↓          │
          │   Residual      │
          └────────┬────────┘
                   │
              × 4 blocks
                   │
                   ↓
             Final LayerNorm
                   │
                   ↓
              LM Head
             64 → vocab
                   │
                   ↓
                LOGITS
                   │
                   ↓
              Cross Entropy
                   │
                   ↓
                  LOSS
                   │
                   ↓
             Backpropagation
                   │
                   ↓
                AdamW
                   │
                   └────→ update weights
```

---

# 38. The most important shapes to memorize

With your configuration:

```python
B = 16
T = 32
C = 64
n_head = 4
head_size = 16
```

Remember this table:

| Object               | Shape                |
| -------------------- | -------------------- |
| `idx`                | `(16,32)`            |
| `tok_emb`            | `(16,32,64)`         |
| `pos_emb`            | `(32,64)`            |
| `x`                  | `(16,32,64)`         |
| `q`                  | `(16,32,16)`         |
| `k`                  | `(16,32,16)`         |
| `v`                  | `(16,32,16)`         |
| `q @ k.T`            | `(16,32,32)`         |
| One head output      | `(16,32,16)`         |
| 4 heads concatenated | `(16,32,64)`         |
| FFN intermediate     | `(16,32,256)`        |
| FFN output           | `(16,32,64)`         |
| logits               | `(16,32,vocab_size)` |

And the **three dimensions** have very clear meanings:

```text
B = Batch
T = Time / sequence positions
C = Channels / embedding dimensions
```

So when you see:

```text
(B,T,C)
```

read it as:

> **16 sequences × 32 positions × 64 features per position**

---

## One final mental model

If you remember nothing else, remember this:

### Attention

```text
Q + K
   ↓
"Who should I listen to?"
   ↓
attention weights
   ↓
weighted V
   ↓
"Here's the information I gathered."
```

### Feed Forward

```text
information gathered by attention
             ↓
          64 → 256
             ↓
          nonlinearity
             ↓
          256 → 64
             ↓
"Here's my transformed representation."
```

### Residual

```text
old representation
       +
new information
       ↓
updated representation
```

### LM Head

```text
updated representation
       ↓
64 numbers
       ↓
vocab_size numbers
       ↓
"What should the next token be?"
```

**That's the entire GPT.**

And one thing I strongly recommend for your next revision: **don't reread this passively.** Take a blank sheet and try to recreate the shape flow from memory:

```text
(B,T)
 ↓
(B,T,C)
 ↓
Q/K/V
 ↓
(B,T,head_size)
 ↓
(B,T,T)
 ↓
(B,T,head_size)
 ↓
(B,T,C)
 ↓
(B,T,4C)
 ↓
(B,T,C)
 ↓
(B,T,vocab_size)
```

If you can explain *why every arrow changes or preserves the shape*, you genuinely understand the implementation rather than just recognizing Karpathy's code.
