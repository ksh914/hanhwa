import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import warnings
from function_file.ML_functions import * 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)
          

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.input_embedding  = nn.Linear(1,feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        # src with shape (input_window, batch_len, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

#######################################################################################
def multistep_time_series(input_data, input_window, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L-input_window):
        train_seq = np.append(input_data[i:i+input_window][:-output_window] , output_window * [0])
        train_label = input_data[i:i+input_window]
        #train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def singlestep_time_series(input_data, input_window, output_window, block_len):
    inout_seq = []
    L = len(input_data)
    block_num =  L - block_len + 1
    # total of [N - block_len + 1] blocks
    # where block_len = input_window + output_window

    for i in range( block_num ):
        train_seq = input_data[i : i + input_window]
        train_label = input_data[i + output_window : i + input_window + output_window]
        inout_seq.append((train_seq ,train_label))

    return torch.FloatTensor(np.array(inout_seq))
#######################################################################################




from function_file.ML_functions import make_dataframe
from function_file.time_series import time_series_dataframe
from sklearn.preprocessing import MinMaxScaler
scaler_train = MinMaxScaler()
scaler_test = MinMaxScaler()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TransAm().to(device)
lr = 0.001
optimizer = torch.optim.Adam(params = model.parameters, lr = lr)
criterion = nn.MSELoss()
epochs = 100
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

def get_batch(data, idx, batch_size):
    seq_len = min(batch_size, len(data) -1 - idx)
    tmp = data[idx : idx + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target

def train(train_data, batch_size):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch, i in enumerate(range(0,len(train_data) -1, batch_size)):
        data,target = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def plot_and_loss(model, source, output_window):
    model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    with torch.no_grad():
        for i in range(0, len(source) -1):
            data, target = get_batch(source, i, 1)
            output = model(data)
            total_loss += criterion(output, target).item()

            test_result = torch.cat((test_result, output[-output_window:].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-output_window:].view(-1).cpu()),0)

    test_result = scaler_test.inverse_transform(test_result.reshape(-1,1)).rehspae(-1)
    truth = scaler_test.inverse_transform(truth.reshape(-1,1)).reshape(-1)

    plt.plot(test_result, color = 'red')
    plt.plot(truth, color='blue')
    plt.grid(True, which = 'both')
    plt.axhline(y=0, color='k')
    plt.ylim([600,900])
    plt.show()
    plt.close()

    return test_result, truth, total_loss / i

def evaluate(model, source):
    model.eval()
    total_loss = 0.0
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(source)-1, batch_size):
            data, targets = get_batch(source, i, batch_size)
            output = model(data)
            total_loss += len(data[0]) * criterion(targets, output).cpu().item()

    return total_loss / len(source)


def train_transformer(input_window, output_window, output_type, batch_size):
    block_len = input_window + output_window
    batch_size = batch_size
    
    df = time_series_dataframe()
    df = df['TEMP'].values
    train_len = int(len(df) * 0.7)
    train_data = df[:train_len]
    test_data = df[train_len:]

    train_data = scaler_train.fit_transform(train_data.reshape(-1,1)).reshape(-1)
    test_data = scaler_test.fit_transform(test_data.reshape(-1,1)).reshape(-1)
    

    if output_type == 'multi':
        train_data = multistep_time_series(train_data, input_window, output_window)
        test_data =  multistep_time_series(test_data, input_window, output_window)

    else:
        train_data = singlestep_time_series(train_data, input_window, output_window,block_len)
        test_data = singlestep_time_series(test_data, input_window, output_window,block_len)
        
    train_data = train_data[:-output_window]
    test_data = test_data[:-output_window]

    train_data = train_data.to(device)
    test_data = test_data.to(device)
    
    for epoch in range(1, epochs + 1):
        
        start_time = time.time()

        train(train_data, batch_size=batch_size)

        if (epochs % 20 == 0):
            test_result, truth, val_loss =  plot_and_loss(model= model, source = test_data, output_window = output_window)

        else:
            val_loss = evaluate(model, test_data)

        print('-' * 90)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f} |'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 90)

        scheduler.step()

    return test_result, truth

