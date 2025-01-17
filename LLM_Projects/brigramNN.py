import torch
import torch.nn.functional as F
#initialize weight matrix
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator = g, requires_grad = True)

words = open('names.txt', 'r').read().splitlines()
N = torch.zeros((27,27), dtype = torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

#training set and target 
xs, ys = [], []

for w in words:  ###
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()


#Gradient descent
while True:
    #Forward pass
    #One Hot encoding
    xenc = F.one_hot(xs, num_classes = 27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims = True)
    #Loss
    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())
    #Backward pass
    W.grad = None
    loss.backward()

    W.data += -50 * W.grad