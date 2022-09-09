from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import tinygrad.nn.optim as optim
import numpy as np

#self.h acts as the hidden state at t-1
class RNN:
    def __init__(self, input_size:int, hidden_size:int, output_size:int, rollout:int, lr:float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rollout = rollout

        self.W_xh = Tensor.uniform(self.input_size, self.hidden_size) 
        self.W_hh = Linear(self.hidden_size, self.hidden_size)
        self.W_hy = Linear(self.hidden_size, self.output_size)

        self.h = Tensor.zeros(hidden_size) #the hidden state that we pass to the next iteration

    def step(self, x):
        self.h = (x.dot(self.W_xh) + self.W_hh(self.h)).tanh()
        print("this:")
        print(self.W_hh(self.h).shape)
        y = self.W_hy(self.h)
        return y

    def resetMemory(self):
        self.h = Tensor.zeros(self.hidden_size)

#make a vector representing this char
def charToTen(char:str):
    return Tensor([0 if x != char_to_idx[char] else 1 for x in range(vocab_size)], requires_grad=False) #zero vector except value 1 at index of this character

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
    print(preds.shape)
    idx = np.random.choice(range(vocab_size), p=preds.ravel())
    return idx_to_char[idx]

raw_data = open('input2.txt', 'r').read()
words = raw_data.split()

def no_remove(x):
    if not (x[0:4] == "http" or x[0] == '<'): return True

trimmed = list(filter(no_remove, words))
data:str = ' '.join(trimmed) #giant string

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

hidden_size =7
seq_len = 2
lr = 1e-1

model = RNN(vocab_size, hidden_size, vocab_size, seq_len, lr)
tens_to_track = [model.W_xh, model.W_hh.weight, model.W_hh.bias, model.W_hy.weight, model.W_hy.bias]
optim = optim.SGD(tens_to_track, lr=lr)

n,p = 0,0 #p is like our point in the data and n is the current iteration
mW_xh, mW_hh, mW_hy = Tensor.zeros(*model.W_xh.shape), Tensor.zeros(*model.W_hh.weight.shape), Tensor.zeros(*model.W_hy.weight.shape) #memory for adagrad
while True:
    if p+seq_len+1 >= len(data) or n==0:
        model.resetMemory()
        p = 0
    inputs = [ch for ch in data[p:p+seq_len]]
    targets = [ch for ch in data[p+1:p+seq_len+1]] #target is the next letter
    #targets.append('a')
    print(inputs)
    print(targets)

    loss = 0
    for x,t in zip(inputs,targets):
        out = model.step(charToTen(x))
        e = (out - charToTen(t)) * Tensor([[1.0]]) #times one is dumb hack to change shape from nx to nx1
        loss += e.dot(e).mean() #mse, dot with self == square
        optim.zero_grad()
        print(loss.deepwalk())
        exit()
        loss.backward()
        optim.step()
    p += seq_len
    n += 1

    exit()
