from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import tinygrad.nn.optim as optim
import numpy as np

#self.h acts as the hidden state at t-1
class RNN():
    def __init__(self, input_size:int, hidden_size:int, output_size:int): 
        self.hidden_size = hidden_size

        self.W_xh = Linear(input_size + hidden_size, hidden_size) 
        #self.W_hh = Linear(hidden_size, hidden_size)
        self.W_hy = Linear(input_size + hidden_size, output_size)

        #self.h = Tensor.zeros(hidden_size) #the hidden state that we pass to the next iteration

    def step(self,x,h):
        combined = x.cat(h, dim=1)
        h = self.W_xh(combined)
        y = self.W_hy(combined)
        y = y.logsoftmax()
        #self.h = (x.dot(self.W_xh) + self.W_hh(self.h)).tanh()
        #y = self.W_hy(self.h)
        return y,h

    def initHidden(self):
        return Tensor.zeros(1,self.hidden_size)

    def resetMemory(self):
        self.h = Tensor.zeros(self.hidden_size)

#make a vector representing this char
def charToTen(char:str):
    return Tensor([0 if x != char_to_idx[char] else 1 for x in range(vocab_size)], requires_grad=False) * Tensor.ones(1,1)#zero vector except value 1 at index of this character

def sample(h:Tensor, seed_letter:str, out_len:int):
    x = charToTen(seed_letter)
    out = ""
    for n in range(out_len):
        y = model.step(x)
        out += outTenToChar(y)
    return out

#take model output and get predicted character
def outTenToChar(out:Tensor):
    preds = np.exp(out.numpy()) / np.sum(np.exp(out.numpy()))
    idx = np.random.choice(range(vocab_size), p=preds.ravel())
    return idx_to_char[idx]

def no_remove(x):
    if not (x[0:4] == "http" or x[0] == '<'): return True

#data
raw_data = open('input2.txt', 'r').read()
words = raw_data.split()
trimmed = list(filter(no_remove, words))
data:str = ' '.join(trimmed) #giant string
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
#maps from char to their respective number reps
char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

#hyperparameters
hidden_size = 128
seq_len = 25
lr = 1e-1

prev_state = Tensor.zeros(hidden_size)

model = RNN(vocab_size, hidden_size, vocab_size)
#tens_to_track = [model.W_xh.weight, model.W_xh.bias, model.W_hh.weight, model.W_hh.bias, model.W_hy.weight, model.W_hy.bias]
tens_to_track = [model.W_xh.weight, model.W_xh.bias, model.W_hy.weight, model.W_hy.bias]
optim = optim.SGD(tens_to_track, lr=lr)

n,p = 0,0 #p is like our point in the data and n is the current iteration
#mW_xh, mW_hh, mW_hy = Tensor.zeros(*model.W_xh.shape), Tensor.zeros(*model.W_hh.weight.shape), Tensor.zeros(*model.W_hy.weight.shape) #memory for adagrad
while True:
    if p+seq_len+1 >= len(data) or n==0:
        model.resetMemory()
        p = 0
    inputs = [ch for ch in data[p:p+seq_len]]
    targets = [ch for ch in data[p+1:p+seq_len+1]] #target is the next letter
    loss = Tensor.zeros(1)
    hidden = model.initHidden()
    current_out = ""
    for x,t in zip(inputs,targets):
        out, hidden = model.step(charToTen(x), hidden)
        current_out += outTenToChar(out)
        #probs = (out/(out.exp().sum())).exp()
        e = (out - charToTen(t)) * Tensor.ones(1,1)
        loss += e.dot(e).mean() #mse, dot with self == square
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"loss at {n}: {loss.numpy()}")
    print(current_out)
    p += seq_len
    n += 1
