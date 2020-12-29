import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rnn.attention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Inputs:
    input : (batch_size, sequence_length_size, input_size)
    (inter_modal_h : (batch_size, hidden_size(hidden_size*2)))
Outputs:
    Bidirectional == False:
        out, (c_out) : (batch_size, sequence_length_size, hidden_size)
        h, (c) : (batch_size, hidden_size) = out[:-1]
    Bidirectional == True:
        out, (c_out) : (batch_size, sequence_length_size, hidden_size*2)
        h, (c) : (batch_size, hidden_size*2) = out[:-1]
"""


class GRU_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.gru)

    def forward(self, input, seq_length=None):
        seq_len = input.size(1)
        h0 = torch.zeros(input.size(0), self.hidden_size).to(input.device)
        h_in = [h0] * self.num_layers
        out = []
        for i in range(seq_len):
            h_i = input[:, i, :]
            for j, layer in enumerate(self.gru):
                h_i = layer(h_i, h_in[j])
                h_in[j] = h_i
            out.append(h_i)
        out = torch.stack(out, dim=1)
        h = torch.stack(h_in, dim=0)
        return out, h


class Bidirectional_GRU_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_f = nn.ModuleList()
        self.gru_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru_f.append(nn.GRUCell(input_size, hidden_size))
            self.gru_b.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.gru_f)
        # print(self.gru_b)

    def reverse(self, x):
        reverse_x = torch.flip(x, [1])
        return reverse_x

    def forward(self, input, seq_length=None):
        seq_len = input.size(1)
        h0 = torch.zeros(input.size(0), self.hidden_size).to(input.device)
        h_f_in = [h0] * self.num_layers
        h_b_in = [h0] * self.num_layers
        reverse_input = self.reverse(input)
        out_f = []
        out_b = []
        for i in range(seq_len):
            h_f = input[:, i, :]
            h_b = reverse_input[:, i, :]
            for j, (layer_f, layer_b) in enumerate(zip(self.gru_f, self.gru_b)):
                h_f = layer_f(h_f, h_f_in[j])
                h_b = layer_b(h_b, h_b_in[j])
                h_f_in[j] = h_f
                h_b_in[j] = h_b
            out_f.append(h_f)
            out_b.append(h_b)
        out_f = torch.stack(out_f, dim=1)
        out_b = torch.stack(out_b, dim=1)
        out_b = self.reverse(out_b)
        out = torch.cat([out_f, out_b], dim=2)
        h_f_out = torch.stack(h_f_in, dim=0)
        h_b_out = torch.stack(h_b_in, dim=0)
        h = torch.cat([h_f_out, h_b_out], dim=2)
        return out, h


class GRU_Encoder_(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.padidx = 0
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )

    def forward(self, input, seq_length=None):
        batch_size = input.size(0)
        if seq_length is not None:
            # input: PackedSequence of (batch_size, seq_length, input_size)
            input = pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        output, h = self.gru(input)
        h = h.reshape(self.num_layers, batch_size, -1)
        # output: (batch_size, seq_length, hidden_size)
        # h: (num_layers, batch_size, hidden_size)
        if seq_length is not None:
            output, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        return output, h


class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.lstm)

    def forward(self, input, seq_length=None):
        seq_len = input.size(1)
        h0 = torch.zeros(input.size(0), self.hidden_size).to(input.device)
        c0 = torch.zeros(input.size(0), self.hidden_size).to(input.device)
        h_in = [h0] * self.num_layers
        c_in = [c0] * self.num_layers
        out = []
        for i in range(seq_len):
            h_i = input[:, i, :]
            for j, layer in enumerate(self.lstm):
                h_i, c_i = layer(h_i, (h_in[j], c_in[j]))
                h_in[j] = h_i
                c_in[j] = c_i
            out.append(h_i)
        out = torch.stack(out, dim=1)
        h = torch.stack(h_in, dim=0)
        c = torch.stack(c_in, dim=0)
        return out, (h, c)


class Bidirectional_LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_f = nn.ModuleList()
        self.lstm_b = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm_f.append(nn.LSTMCell(input_size, hidden_size))
            self.lstm_b.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.lstm_f)
        # print(self.lstm_b)

    def reverse(self, x):
        reverse_x = torch.flip(x, [1])
        return reverse_x

    def forward(self, input, inter_modal_h=None):
        seq_len = input.size(1)
        h0 = torch.zeros(input.size(0), self.hidden_size).to(input.device)
        c0 = torch.zeros(input.size(0), self.hidden_size).to(input.device)
        h_f_in = [h0] * self.num_layers
        h_b_in = [h0] * self.num_layers
        c_f_in = [c0] * self.num_layers
        c_b_in = [c0] * self.num_layers
        reverse_input = self.reverse(input)
        out_f = []
        out_b = []
        for i in range(seq_len):
            h_f = input[:, i, :]
            h_b = reverse_input[:, i, :]
            for j, (layer_f, layer_b) in enumerate(zip(self.lstm_f, self.lstm_b)):
                h_f, c_f = layer_f(h_f, (h_f_in[j], c_f_in[j]))
                h_b, c_b = layer_b(h_b, (h_b_in[j], c_b_in[j]))
                h_f_in[j] = h_f
                h_b_in[j] = h_b
                c_f_in[j] = c_f
                c_b_in[j] = c_b
            out_f.append(h_f)
            out_b.append(h_b)
        out_f = torch.stack(out_f, dim=1)
        out_b = torch.stack(out_b, dim=1)
        out_b = self.reverse(out_b)
        out = torch.cat([out_f, out_b], dim=2)
        h_f_out = torch.stack(h_f_in, dim=0)
        h_b_out = torch.stack(h_b_in, dim=0)
        c_f_out = torch.stack(h_f_in, dim=0)
        c_b_out = torch.stack(h_b_in, dim=0)
        h = torch.cat([h_f_out, h_b_out], dim=2)
        c = torch.cat([c_f_out, c_b_out], dim=2)
        return out, (h, c)


class LSTM_Encoder_(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        if bidirectional:
            self.hidden_size = hidden_size * 2
        else:
            self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padidx = 0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )

    def forward(self, input, seq_length=None):
        batch_size = input.size(0)
        if seq_length is not None:
            # input: PackedSequence of (batch_size, seq_length, input_size)
            input = pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(input)
        h = h.reshape(self.num_layers, batch_size, -1)
        c = c.reshape(self.num_layers, batch_size, -1)
        # output: (batch_size, seq_length, self.hidden_size)
        # h, c: (num_layers, batch_size, self.hidden_size)
        if seq_length is not None:
            out, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        return output, (h, c)