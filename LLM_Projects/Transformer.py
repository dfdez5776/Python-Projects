#GPT From Scratch
#Study of "Attention is All You Need"
import torch
import torch.nn as nn
from torch.nn import functional as F
from encoder import *




#Hyperparameters
batch_size = 32 #Sequences to process in parallel
block_size = 8 #context length for predictions
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
torch.manual_seed(1337)
n_embd = 32

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
image = []

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
class Head(nn.Module):
     #One head of self- attention (self attention for encoders)

     def __init__(self, head_size):
          super().__init__()
          self.key = nn.Linear(n_embd, head_size, bias = False)
          self.query = nn.Linear(n_embd, head_size, bias = False)
          self.value = nn.Linear(n_embd, head_size, bias = False)
          self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

          self.e_key = nn.Linear(n_embd, head_size, bias = False)
          self.e_query = nn.Linear(n_embd, head_size, bias = False)
          self.e_value = nn.Linear(n_embd, head_size, bias = False)
          
     def forward(self, x, input):
        B,T, C = x.shape
        #A single head
        k = self.key(x) #B,T,8
        q = self.query(x)  #B,T,8
        
        wei = q @ k.transpose(-2, -1) # (B, T, 8) @ (B, 8, T) ---> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei =F.softmax(wei, dim = -1)
        v = self.value(x)


        #Encoder 
        e_k = self.e_key(input)
        e_q = self.e_query(input)
        e_wei = e_q @ e_k.transpose(-2, -1)
        e_wei = F.softmax(wei, dim = -1)

        e_v = self.e_value(input)
        e_out = e_wei @ e_v


        print(v.shape)

        out = wei @ v # (b,T, T) @ (B, T,C) ---> (B,T,C)


        #cross attention
        #output of encoder multiplied by encoder key matrix,
        cross_wei = self.query(x)@self.e_key(e_out).transpose(-2, -2)
        cross_wei = F.softmax(wei, dim = -1)
        cross_v = self.e_value(e_out)

        cross_out = cross_wei @ cross_v

        out = out + cross_out



        return out



class MultiHeadAttention(nn.Module):
     
     def __init__(self, num_heads, head_size):
          super().__init__()
          self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
          self.proj = nn.Linear(n_embd, n_embd)

     def forward(self, x):
          out = torch.cat([h(x) for h in self.heads], dim = -1)
          out = self.proj(out)
          return out

#Feed Forward 
class FeedForward(nn.Module):
     
    def __init__(self, n_embd):
         super().__init__()
         self.net = nn.Sequential(
              nn.Linear(n_embd, 4* n_embd),
              nn.ReLU(),
              nn.Linear(4*n_embd, n_embd),

         )
    def forward(self, x):
         return self.net(x)

class Block(nn.Module):
     
     def __init__(self, n_embd, n_head):
         super().__init__()
         head_size = n_embd // n_head
         self.sa = MultiHeadAttention(n_head, head_size)
         self.ffwd = FeedForward(n_embd)
         self.ln1 = nn.LayerNorm(n_embd)
         self.ln2 = nn.LayerNorm(n_embd)

         self.e_self_attention = MultiHeadAttention(n_head, head_size)
         self.e_feedforward = FeedForward(n_embd)
         self.e_llnorm1 = nn.LayerNorm(n_embd)
         self.e_llnorm2 = nn.LayerNorm(n_embd)


         

     def forward(self, x):
         x = x + self.sa(self.ln1(x), self.e_llnorm1(input))
         x = x + self.ffwd(self.ln2(x))
         return x



#Define Bigram Model

class BigramLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(
                 Block(n_embd, n_head = 4),
                 Block(n_embd, n_head = 4),
                 Block(n_embd, n_head = 4),
                 nn.LayerNorm(n_embd))

            #Neural Nets from encoder architecture
            self.e_blocks = nn.Sequential(
                blocks(n_embd, n_head = 4),
                blocks(n_embd, n_head = 4),
                blocks(n_embd, n_head = 4),
                nn.LayerNorm(n_embd)


            )

            self.lm_head = nn.Linear(n_embd, vocab_size)
            self.e_lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, input, targets=None):
             B, T = idx.shape
             #idx and targets are both (B,T) tensor of integers
             tok_emb = self.token_embedding_table(idx) #(BTC)
             e_tok_emb = self.ImageEmbedding(input)
             pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)

             e_x = e_tok_emb
             x = tok_emb + pos_emb

             e_x = self.blocks(e_x)
             x = self.blocks(x)

             e_logits = self.e_lm_head(e_x)
             logits = self.lm_head(x) #(BT, Vocab size)

             if targets is None:
                  loss = None
             else:
                B, T, C = logits.shape
                e_logits = logits.view(B*T, C)
                logits = logits.view(B*T, C)
                
                e_targets = targets.view(B*T)
                targets = targets.view(B*T)
                #Same as Negative loss likelihood
                e_loss = F.cross_entropy(e_logits, e_targets)
                loss = F.cross_entropy(logits, targets)
             
             return logits, loss, e_logits, e_loss
        
        def generate(self, idx, max_new_tokens):
             # idx its (B,T) array of indices in the current context
             for _ in range(max_new_tokens):
                    print("generating")
                    #crop idc 
                    idx_cond = idx[:, -block_size:]
                    #get the predictions
                    logits, loss = self(idx_cond)
                    #focus on last time step
                    logits = logits[:, -1, :] #becomes B,c
                    #apply softmax to get probabilities
                    probs = F.softmax(logits, dim = -1) #(B,C)
                    #sample from distributions
                    idx_next = torch.multinomial(probs, num_samples = 1) #(B,1)
                    #append sampled index to running sequences
                    idx = torch.cat((idx, idx_next), dim =1) #(B,T+1)
             return idx
        
m = BigramLanguageModel()
#endoder

m = m.to(device)

logits, loss, e_logits, e_loss = m(xb, yb, image)


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
    logits , loss = m(xb, yb, image)
    print('xb' , xb.shape)
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1,1), dtype=torch.long, device = device)
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens = 500)[0].tolist()))


