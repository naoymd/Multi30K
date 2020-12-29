import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from rnn.attention import Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Inputs:
    Bidirectinal == False:
        input : (batch_size, sequence_length_size, input_size)
        encoder_out : (batch_size, sequence_length_size, hidden_size)
        encoder_h : (batch_size, hidden_size)
        (encoder_out : (batch_size, sequence_length_size, hidden_size(hidden_size*2)))
    Bidirectinal == True:
        input : (batch_size, sequence_length_size, input_size)
        encoder_out : (batch_size, sequence_length_size, hidden_size)
        encoder_h : (batch_size, hidden_size)
        (encoder_out : (batch_size, sequence_length_size, hidden_size(hidden_size*2)))
Outputs:
    Bidirectional == False:
        out, (c_out) : (batch_size, sequence_length_size, hidden_size)
        h, (c) : (batch_size, hidden_size) = out[:-1]
    Bidirectional == True:
        out, (c_out) : (batch_size, sequence_length_size, hidden_size*2)
        h, (c) : (batch_size, hidden_size*2) = out[:-1]
"""

class GRU_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_attention = Attention(hidden_size)
        self.gru = nn.ModuleList()
        for i in range(self.num_layers):
            self.gru.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.gru)

    def forward(self, input, encoder_h, encoder_out=None, seq_length=None):
        seq_len = input.size(1)
        if encoder_out is not None:
            encoder_h, _ = self.h_attention(encoder_h.permute(1, 0, 2), encoder_out)
            encoder_h = encoder_h.permute(1, 0, 2)
        h_in = [encoder_h[i, :, :] for i in range(self.num_layers)]
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


class Bidirectional_GRU_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_attention = Attention(hidden_size*2)
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

    def forward(self, input, encoder_h, encoder_out=None, seq_length=None):
        seq_len = input.size(1)
        if encoder_out is not None:
            encoder_h, _ = self.h_attention(encoder_h.permute(1, 0, 2), encoder_out)
            encoder_h = encoder_h.permute(1, 0, 2)
        h_f_in = [encoder_h[i, :, :self.hidden_size] for i in range(self.num_layers)]
        h_b_in = [encoder_h[i, :, self.hidden_size:] for i in range(self.num_layers)]
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


class GRU_Decoder_(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padidx = 0
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )
        self.rnn_attention = Attention(hidden_size, output_mode='concat')

    def forward(self, input, encoder_h, encoder_out=None, seq_length=None):
        batch_size = input.size(0)
        encoder_h = encoder_h.reshape(-1, batch_size, self.hidden_size)
        if seq_length is not None:
            # input: PackedSequence of (batch_size, seq_length, input_size)
            input = pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        output, h = self.gru(input, encoder_h)
        h = h.reshape(self.num_layers, batch_size, -1)
        # output: (batch_size, seq_length, hidden_size)
        # h: (num_layers, batch_size, hidden_size)
        if seq_length is not None:
            out, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        if encoder_out is not None:
            output, _ = self.rnn_attention(output, encoder_out)
        return output, h


class GRU_Skip_Decoder_(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padidx = 0
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )
        if bidirectional:
            hidden_size = hidden_size * 2
        else:
            hidden_size = hidden_size
        self.rnn_attention = Attention(hidden_size, output_mode='concat')
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, input, encoder_h, encoder_out=None, seq_length=None):
        batch_size = input.size(0)
        encoder_h = encoder_h.reshape(-1, batch_size, self.hidden_size)
        if seq_length is not None:
            # input: PackedSequence of (batch_size, seq_length, input_size)
            input = pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        output, h = self.gru(input, encoder_h)
        h = h.reshape(self.num_layers, input_size, -1)
        # output: (batch_size, seq_length, hidden_size)
        # h: (num_layers, batch_size, hidden_size)
        if seq_length is not None:
            out, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        if encoder_out is not None:
            output, _ = self.rnn_attention(output, encoder_out)
        output += self.linear(input)
        return output, h


class LSTM_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_attention = Attention(hidden_size)
        self.c_attention = Attention(hidden_size)
        self.lstm = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstm.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size
        # print(self.lstm)

    def forward(self, input, encoder_h, encoder_out=None, seq_length=None):
        seq_len = input.size(1)
        encoder_h, encoder_c = encoder_h
        if encoder_out is not None:
            encoder_h, _ = self.h_attention(encoder_h.permute(1, 0, 2), encoder_out)
            # encoder_c, _ = self.c_attention(encoder_c.permute(1, 0, 2), encoder_out)
            encoder_h = encoder_h.permute(1, 0, 2)
            # encoder_c = encoder_c.permute(1, 0, 2)
        h_in = [encoder_h[i, :, :] for i in range(self.num_layers)]
        c_in = [encoder_c[i, :, :] for i in range(self.num_layers)]
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


class Bidirectional_LSTM_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h_attention = Attention(hidden_size*2)
        self.c_attention = Attention(hidden_size*2)
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

    def forward(self, input, encoder_h, encoder_out=None, seq_length=None):
        seq_len = input.size(1)
        encoder_h, encoder_c = encoder_h
        if encoder_out is not None:
            encoder_h, _ = self.h_attention(encoder_h.permute(1, 0, 2), encoder_out)
            # encoder_c, _ = self.c_attention(encoder_c.permute(1, 0, 2), encoder_out)
            encoder_h = encoder_h.permute(1, 0, 2)
            # encoder_c = encoder_c.permute(1, 0, 2)
        h_f_in = [encoder_h[i, :, :self.hidden_size] for i in range(self.num_layers)]
        h_b_in = [encoder_h[i, :, self.hidden_size:] for i in range(self.num_layers)]
        c_f_in = [encoder_c[i, :, :self.hidden_size] for i in range(self.num_layers)]
        c_b_in = [encoder_c[i, :, self.hidden_size:] for i in range(self.num_layers)]
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
        c_f_out = torch.stack(c_f_in, dim=0)
        c_b_out = torch.stack(c_b_in, dim=0)
        h = torch.cat([h_f_out, h_b_out], dim=2)
        c = torch.cat([c_f_out, c_b_out], dim=2)
        return out, (h, c)


class LSTM_Decoder_(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
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
        if bidirectional:
            hidden_size = hidden_size * 2
        else:
            hidden_size = hidden_size
        self.rnn_attention(hidden_size, output_mode='concat')

    def forward(self, input, encoder_h, encoder_out=None, seq_length=None):
        batch_size = input.size(0)
        encoder_h, encoder_c = encoder_h
        encoder_h = encoder_h.reshape(-1, batch_size, self.hidden_size)
        encoder_c = encoder_c.reshape(-1, batch_size, self.hidden_size)
        if seq_length is not None:
            # input: PackedSequence of (batch_size, seq_length, input_size)
            input = pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(input, (encoder_h, encoder_c))
        h = h.reshape(self.num_layers, batch_size, -1)
        c = c.reshape(self.num_layers, batch_size, -1)
        # output: (batch_size, seq_length, hidden_size)
        # h, c: (num_layers, batch_size, hidden_size)
        if seq_length is not None:
            out, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        if encoder_out is not None:
            output, _ = self.rnn_attention(output, encoder_out)
        return output, (h, c)


class LSTM_Skip_Decoder_(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
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
        if bidirectional:
            hidden_size = hidden_size * 2
        else:
            hidden_size = hidden_size
        self.rnn_attention(hidden_size, output_mode='concat')
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, input, encoder_h, encoder_out, seq_length=None):
        batch_size = input.size(0)
        encoder_h, encoder_c = encoder_h
        encoder_h = encoder_h.reshape(-1, batch_size, self.hidden_size)
        encoder_c = encoder_c.reshape(-1, batch_size, self.hidden_size)
        if seq_length is not None:
            # input: PackedSequence of (batch_size, seq_length, input_size)
            input = pack_padded_sequence(input, seq_length, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(input, (encoder_h, encoder_c))
        h = h.reshape(self.num_layers, batch_size, -1)
        c = c.reshape(self.num_layers, batch_size, -1)
        # output: (batch_size, seq_length, hidden_size)
        # h, c: (num_layers, batch_size, hidden_size)
        if seq_length is not None:
            out, lengths = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)
        if encoder_out is not None:
            output, _ = self.rnn_attention(output, encoder_out)
        output += self.linear(input)
        return output, (h, c)