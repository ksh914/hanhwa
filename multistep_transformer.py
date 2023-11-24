import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from ML_functions import make_dataframe

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

calculate_loss_over_all_values = False

# S : source sequence length(lookback size)
# T : target sequence length(output size)
# N : Batch size
# E : feature number

# src = torch.rand((10,32,512)) # (S, N, E)
# tgt = torch.rand((20,32,512)) # (T, N, E)
# out = transformer_model(src,tgt)
# 
# print(out)

input_window = 100
output_window = 5
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)

        # positions shape : [max_len, 1]
        positions = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) 

        # div_term shape : [d_model / 2]
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)) / d_model)

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        # Saving buffer(same as parameter without gradients needed)
        pe = pe.unsqueeze(0).transpose(0,1)

        #pe.requires_grad = False
        self.register_buffer('pe',pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class Transformer(nn.Module):

    def __init__(self, feature_size = 250, num_layers = 1, dropout = 0.1):
        super(Transformer, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = feature_size, nhead = 10, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
        self.decoder = nn.Linear(feature_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).maksed_fill(mask == 1, float(0.0))
        return mask
    
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)

    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window], output_window * [0])
        train_label = input_data[i:i+tw]
        inout_seq.append((train_seq, train_label))

    return torch.FloatTensor(inout_seq)

class TransformerDataset(Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]
        

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


    def __len__(self):
        return len(self.X)
    
def get_data(df):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    data = scaler.fit_transform(df['TEMP'].reshape(-1,1)).reshape(-1)

    train_len = int(len(data) * 0.9)
    train_data = data[:train_len]
    test_data = data[train_len:]

    train_seq = create_inout_sequences(train_data, input_window)
    train_seq = train_seq[:-output_window]

    test_seq = create_inout_sequences(test_data, input_window)
    test_seq = test_seq[:-output_window]

    return train_seq.to(device), test_seq.to(device)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) -1 -i)
    data = source[i : i+seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target

def train(train_data):
    model.train()
    total_loss = 0.
    start_time = time.time()
    
    for batch, i in enumerate(range(0, len(train_data)-1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output,targets)

        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time

            print('| epoch : {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                                                      elapsed * 1000 / log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    with torch.no_grad():
        for i in range(0, len(data_source)-1):
            data,target = get_batch(data_source, i, 1)
            output = eval_model(data)

            if calculate_loss_over_all_values:
                total_loss += criterion(output,target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:])

            test_result = torch.cat((test_result, output[-1].view(-1).cpu()),0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()),0)

    
    len(test_result)

    plt.plot(test_result, color='red')
    plt.plot(truth[:500], color='blue')
    plt.plot(test_result - truth, color='green')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.cloe()

    return total_loss / i


def predict_future(eval_model, data_source,steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    _ , data = get_batch(data_source, 0,1)
    with torch.no_grad():
        for i in range(0, steps,1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
            
    data = data.cpu().view(-1)
    

    plt.plot(data,color="red")       
    plt.plot(data[:input_window],color="blue")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.close()
        
# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich 
# auch zu denen der predict_future
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            if calculate_loss_over_all_values:
                total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:                                
                total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()            
    return total_loss / len(data_source)
# def data_preprocessing(dataset):

#     scaler = MinMaxScaler()
#     dataset = scaler.fit_transform(dataset.reshape(-1,1)).reshape(-1)

#     train_len = int(len(dataset) * 0.1)
#     train_data = dataset[:train_len]
#     test_data = dataset[train_len:]
    
#     train_data = create_inout_sequences(train_data, input_window)
#     test_data = create_inout_sequences(test_data, input_window)

#     return train_data[:-output_window], test_data[:-output_window]



model = Transformer().to(device)
criterion = nn.MSELoss()
lr = 0.001
epochs = 10
optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.98)

scaler = MinMaxScaler()

_, dataset = make_dataframe(60,20)

#train_data, test_data = data_preprocessing(dataset)

# train_dataset = TransformerDataset(train_data)
# test_dataset = TransformerDataset(test_data)
# train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
# test_dataloder = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

best_val_loss = float('inf')
best_model = None
train_data, val_data = get_data()

for epoch in range(1, epochs + 1):
    
    epoch_start_time = time.time()
    train(train_data)

    if (epoch % 10 == 0):
        val_loss = plot_and_loss(model, val_data, epoch)
        predict_future(model, val_data, 200)
    
    else:
        val_loss = evaluate(model, val_data)

    print("-" * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    
    scheduler.step() 