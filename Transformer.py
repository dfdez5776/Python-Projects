#GPT From Scratch
#Study of "Attention is All You Need"
import torch
import torch.nn as nn
from torch.nn import functional as F


#Hyperparameters
batch_size = 32 #Sequences to process in parallel
block_size = 8 #context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
torch.manual_seed(1337)

#Download tiny shakespeare, open and read
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))


#sort characters and find unique ones
chars = sorted(list(set(text))) 
vocab_size = len(chars)

#tokenize input text
#create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#could also use SentencePiece modeule by Google, or Tiktoken module by OpenAIi

#Tokenize Entire Library, convert into data tensor
data = torch.tensor(encode(text), dtype = torch.long)

#Split into training/testing

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#Define training blocks
#Maximum context length
block_size = 8
train_data[:block_size+1]

#Batch Dimensions
#set seed
torch.manual_seed(1337)
#Batches are sequences to process in parallel
batch_size = 4
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = m(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

#Generate a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #in case of GPU
    x, y = x.to(device), y.to(device)
    return x,y

xb, yb = get_batch('train')

for b in range(batch_size): #batch dimension
    for t in range(block_size): #time dimension
        context = xb[b, :t+1]
        target = yb[b, t]

#Define Bigram Model

class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, idx, targets=None):
             
             #idx and targets are both (B,T) tensor of integers
             logits = self.token_embedding_table(idx) #(BTC)

             if targets is None:
                  loss = None
             else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                #Same as Negative loss likelihood
                loss = F.cross_entropy(logits, targets)
             
             return logits, loss
        
        def generate(self, idx, max_new_tokens):
             # idx its (B,T) array of indices in the current context
             for _ in range(max_new_tokens):
                    #get the predictions
                    logits, loss = self(idx)
                    #focus on last time step
                    logits = logits[:, -1, :] #becomes B,c
                    #apply softmax to get probabilities
                    probs = F.softmax(logits, dim = -1) #(B,C)
                    #sample from distributions
                    idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
                    #append sampled index to running sequences
                    idx = torch.cat((idx, idx_next), dim =1) #(B,T+1)
             return idx
        
m = BigramLanguageModel(vocab_size)
m = m.to(device)
logits, loss = m(xb, yb)

idx = torch.zeros((1,1), dtype = torch.long)
#print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens = 100)[0].tolist()))
    
#Training the model
#Create a Pytorch optimizer (more advanced than SGD)

optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)
batch_size = 32

for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #sample a batch of data
    losses = estimate_loss()
    xb, yb = get_batch('train')

    #evaluate the loss 
    logits , loss = m(xb, yb)
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens = 500)[0].tolist()))