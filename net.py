import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from rnn.encoder import *
from rnn.decoder import *
from rnn.attention import *
from rnn.transformer import *

def l2norm(out):
  norm = torch.norm(out, dim=-1, keepdim=True)
  return out / norm

class Net(nn.Module):
    def __init__(self, vocab_size, kwargs):
        super(Net, self).__init__()
        self.rnn = kwargs.get('rnn', 'GRU')
        embed_size = kwargs.get('embed_size', 100)
        hidden_size = kwargs.get('hidden_size', 128)
        num_layers = kwargs.get('num_layers', 1)
        bidirection = kwargs.get('bidirection', False)
        self.self_attention = kwargs.get('self_attention', False)
        skip = kwargs.get('skip', False)
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        if self.rnn == 'GRU':
            if bidirection:
                self.encoder = Bidirectional_GRU_Encoder(embed_size, hidden_size, num_layers)
                self.decoder = Bidirectional_GRU_Decoder(embed_size, hidden_size, num_layers)
            else:
                self.encoder = GRU_Encoder(embed_size, hidden_size, num_layers)
                self.decoder = GRU_Decoder(embed_size, hidden_size, num_layers)
        elif self.rnn == 'GRU_':
            if skip:
                self.encoder = GRU_Encoder_(embed_size, hidden_size, num_layers, bidirection)
                self.decoder = GRU_Skip_Decoder_(embed_size, hidden_size, num_layers, bidirection)
            else:
                self.encoder = GRU_Encoder_(embed_size, hidden_size, num_layers, bidirection)
                self.decoder = GRU_Decoder_(embed_size, hidden_size, num_layers, bidirection)
        elif self.rnn == 'LSTM':
            if bidirection:
                self.encoder = Bidirectional_LSTM_Encoder(embed_size, hidden_size, num_layers)
                self.decoder = Bidirectional_LSTM_Decoder(embed_size, hidden_size, num_layers)
            else:
                self.encoder = LSTM_Encoder(embed_size, hidden_size, num_layers)
                self.decoder = LSTM_Decoder(embed_size, hidden_size, num_layers)
        elif self.rnn == 'LSTM_':
            if skip:
                self.encoder = LSTM_Encoder_(embed_size, hidden_size, num_layers, bidirection)
                self.decoder = LSTM_Skip_Decoder_(embed_size, hidden_size, num_layers, bidirection)
            else:
                self.encoder = LSTM_Encoder_(embed_size, hidden_size, num_layers, bidirection)
                self.decoder = LSTM_Decoder_(embed_size, hidden_size, num_layers, bidirection)
        elif self.rnn == 'Transformer':
            self.transformer = Transformer(embed_size, hidden_size, num_layers)
            self.pool = nn.AdaptiveAvgPool1d(1)

        if bidirection:
            self.attention = Attention(hidden_size*2)
            self.fc = nn.Linear(hidden_size*2, vocab_size)
        else:
            self.attention = Attention(hidden_size)
            self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(p=kwargs.get('dropout', 0.5))
        init.xavier_uniform_(self.fc.weight)

    def forward(self, story, query):
        s_embed = self.embed(story)
        q_embed = self.embed(query)

        if 'GRU' in self.rnn:
            en_out, h = self.encoder(s_embed)
            de_out, h = self.decoder(q_embed, h, en_out)
        elif 'LSTM' in self.rnn:
            en_out, h  = self.encoder(s_embed)
            de_out, h = self.decoder(q_embed, h, en_out)
            h, _ = h
        elif 'Transformer' in self.rnn:
            en_out, de_out = self.transformer(s_embed, q_embed)
            h = self.pool(de_out.permute(0, 2, 1)).permute(2, 0, 1)
        
        output = h[-1, :, :]
        if self.self_attention:
            output, attention_map = self.attention(output, en_out)
        output = self.dropout(self.fc(output))
        # output = F.softmax(self.dropout(self.fc(output), dim=-1))
        if self.self_attention:
            return output, attention_map
        else:
            return output