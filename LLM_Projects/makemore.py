import torch

#from visualizecharcounts import *
#Implementation of Bigram Language Model (Next Character Prediction)
#Working with 2 characters at a time. Given one character, predict the next
words = open('names.txt', 'r').read().splitlines()
words[:10]
min(len(w) for w in words)
max(len(w) for w in words)


#Sort information of counts in 2D array. row is first characters, column is 2nd character. Entries record how many occurances happen
#28x28 since there are 26 letters + 1 start + 1 end
N = torch.zeros((27,27), dtype = torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


#tensors allow us to manipulate data very efficiently

#Dictionary to maintain counts of pairs, or bigrams
for w in words:
    #add start and end token to have all information
    chs = ['.'] + list(w) + ['.']
    #iterate through characters
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


P = (N+1).float() #smoothing 
#divide all rows by respective sums
P = P / P.sum(1, keepdim=True)



g = torch.Generator().manual_seed(214748364)

for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]

        ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

#Loss Function (Negative Log Likelihood), find probability model assigns to each bigram vs actual probability
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n +=1
nll = -log_likelihood/n













