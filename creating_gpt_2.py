# -*- coding: utf-8 -*-
"""Creating GPT-2.ipynb

"""

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#The lines below indicate that the text we pulled is ~1 million characters in length. This should give us a pretty good base when we start generating text.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print("How long is the dataset in characters?", len(text))

print(text[:1000])

#Lets sort all the unique characters in the text. This will tell us if there are any unique characters outside of the standard english selection.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#Lets map characters to integers. The idea behind this is to allow us to map the words to tokens or integers which will allow us to predict better.

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # This is where we are creating the encoder to transfer the string input into integers
decode = lambda l: ''.join(itos[i] for i in l) # This creates the decoder to transfer back the encoded text to characters

print(encode("Whats going on?"))
print(decode(encode("Whats going on?")))

# We are going to encode the dataset of Shakespere that we previously downloaded.
# The operations below encode the text into a 64 bit integer within the torch function
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:2000])

#Using the code below, we are going to split the dataset into a test/train. Remember we've converted it to integers and no longer strings/chars
n = int(0.9*len(data)) #90% of the dataset will be used to train. The remaining 10% will be the test/validation set
train_data = data[:n]
validation_data = data[n:]

# Using the block below, we are going to train the model for prediction.
# It's computationally prohibitive to try and do this all at once with the entire text so we are going to break it into block sizes to slowly feed the model.
block_size = 9
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} the target: {target}")

# We previously mentioned batching or partioning the data due to its size and computational needs.
# The script below will batch that data to slowly feed the model.
import torch
torch.manual_seed(1337)
batch_size = 5 # The number of batches we will run in parallel
block_size = 9 # The block size

def get_batch(split):
    # generate a small batch of data of inputs x and y
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('----')

for b in range(batch_size): # The dimensions of the batch
  for t in range(block_size): # time dimensions
    context = xb[b, :t+1]
    target = yb[b,t]
    print(f"When input is {context.tolist()} the target: {target}")

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch,Time,Channel tensor)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #create a 2 dimentional array
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#Lets make the PyTorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(100000):

  # sample a batch of data
  xb, yb = get_batch('train')

  # Lets check out the loss on the sample
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())

# The line below allows us to print the model in terms of tokens. Now it won't match the flow of Shakespere but I can control the output by selecting the max number of new tokens which will give us the output length.
# It also uses the very last character to predict what comes next and not any context. We are going to kick off the transformers to give context.
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

# Mathmatical trick in Self-Attention

torch.manual_seed(1820)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

# Our target goal is to have the tokens talk to each other in terms of context. Now that being said, we can't pull from the future because we are predicting the future.
# The easiest way for tokens to communicate is to do an average of all of the proceeding elements. Ex. 5th token - Avg of 1-4 tokens -- Extremely poor as it has ALOT of loss.

# We want x[b,t] = mean_{i<=t} x[b,i]

xbow = torch.zeros((B,T,C)) #xbow ==> Bag of words on each token
for b in range(B): # Iterating over Batch
    for t in range(T): # Iterating over Time
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

xbow[0]

# The iteration isn't very efficient. If this is done over matrix multiplication, it can be done much more efficiently.
# The code below will do this in a matrix format

torch.manual_seed(1820)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a==')
print(a)
print('--')
print('b==')
print(b)
print('--')
print('c==)')
print(c)

# Batch matrix multiplication to get weighted sums that are in a triangular form. In other words, the tokens in the nth position, only get data from the tokens before it.
weights = torch.tril(torch.ones(T, T))  #lower trangular
weights = weights / weights.sum(1, keepdim=True)
weights
xbow2 = weights @ x # (T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)
xbow[0], xbow2[0]

# Softmax Version (Generalized logistic distribution) We are also setting how the tokens receive data from the previous tokens.
tril = torch.tril(torch.ones(T, T)) #lower trangular
weights = torch.zeros((T,T))  # We are setting the tokens to zero and then the values will become data dependent. The tokens will then see each other for who is "more interesting"
weights = weights.masked_fill(tril == 0, float('-inf')) #all elements where tril is 0 --> Label as -inf
weights = F.softmax(weights, dim=-1)
weights
#You can do weighted averages in a lower trangular fashion of your past elements with matrix multiplication

# Self-Attention Model
torch.manual_seed(1820)
B,T,C = 4,8,32 # Batch, Time, Channels
x = torch.randn(B,T,C)

#single head of self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
weights = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T)) #lower trangular
#weights = torch.zeros((T,T))  # We are setting the tokens to zero and then the values will become data dependent.
weights = weights.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(weights, dim=-1)
out = weights @ x #Think about X as private information for the token itself. This is kept in vector X
v = value(x)
out = weights @ v
out.shape

k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5

k.var()

q.var()

wei.var()

torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)

torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1)

class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

torch.manual_seed(1337)
module = LayerNorm1d(100)
x = torch.randn(32, 100) # batch size 32 of 100-dimensional vectors
x = module(x)
x.shape

x[:,0].mean(), x[:,0].std() # mean,std of one feature across all batch inputs

x[0,:].mean(), x[0,:].std() # mean,std of a single input from the batch, of its features
