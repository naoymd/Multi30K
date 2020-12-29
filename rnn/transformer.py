import math
import torch
import torch.nn as nn
import torch.nn.init as init


class Positional_Encoding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        max_len = 5000
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, dimension)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # pe.size() -> [1, max_len, dimension]
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        input: (batch_size, sequence_length, dimension)
        output: (batch_size, sequence_length, dimension)
        """
        output = input + self.pe[:, :input.size(1), :]
        output = self.dropout(output)
        return output



class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=6, **kwargs):
        super().__init__()
        nhead = kwargs.get('nhead', 8)

        self.src_fc = nn.Linear(input_size, hidden_size)
        self.src_pe = Positional_Encoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # self.encoder = nn.Transformer(hidden_size)

        self.tgt_fc = nn.Linear(input_size, hidden_size)
        self.tgt_pe = Positional_Encoding(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        

    def forward(self, src, tgt=None):
        """
        Inputs:
            src : (batch_size, source_sequence_length, input_size)
            -> TransformerのEncoder側の入力
            tgt : (batch_size, target_sequence_length, input_size)
            -> TransformerのDecoder側の入力
        Outputs:
            output : (batch_size, target_sequence_length, hidden_size)
            -> Transformerの最終出力
        """
        if tgt is None:
            tgt = src
        
        src = self.src_fc(src)
        """ src : (batch_size, source_sequence_length, input_size) """ 
        src_pe = self.src_pe(src).permute(1, 0, 2)
        """ src_pe : (source_sequence_length, batch_size, hidden_size) """        
        src_output = self.encoder(src_pe)
        """ src_output : (source_sequence_length, batch_size, hidden_size) """

        tgt = self.tgt_fc(tgt)
        """ tgt : (batch_size, target_sequence_length, input_size) """
        tgt_pe = self.tgt_pe(tgt).permute(1, 0, 2)
        """ tgt_pe : (target_sequence_length, batch_size, hidden_size) """
        tgt_output = self.decoder(tgt_pe, src_output)
        """ tgt_output : (target_sequence_length, batch_size, hidden_size) """
        tgt_output = tgt_output.permute(1, 0, 2)
        """ tgt_output : (batch_size, target_sequence_length, hidden_size) """
        return src_output, tgt_output



class Transformer_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=6, **kwargs):
        super().__init__()
        nhead = kwargs.get('nhead', 8)

        self.src_fc = nn.Linear(input_size, hidden_size)
        self.src_pe = Positional_Encoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # self.encoder = nn.Transformer(hidden_size)
                

    def forward(self, src):
        """
        Inputs:
            src : (batch_size, source_sequence_length, input_size)
            -> TransformerのEncoder側の入力
        Outputs:
            output : (batch_size, source_sequence_length, hidden_size)
            -> Transformerの最終出力
        """
        
        src = self.src_fc(src)
        """ src : (batch_size, source_sequence_length, input_size) """ 
        src_pe = self.src_pe(src).permute(1, 0, 2)
        """ src_pe : (source_sequence_length, batch_size, hidden_size) """        
        output = self.encoder(src_pe)
        """ output : (source_sequence_length, batch_size, hidden_size) """
        output = output.permute(1, 0, 2)
        """ output : (batch_size, source_sequence_length, hidden_size) """
        return output



class Transformer_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=6, **kwargs):
        super().__init__()
        nhead = kwargs.get('nhead', 8)

        self.tgt_fc = nn.Linear(input_size, hidden_size)
        self.tgt_pe = Positional_Encoding(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        

    def forward(self, src, tgt):
        """
        Inputs:
            src : (batch_size, source_sequence_length, hidden_size)
            -> TransformerのEncoder側の出力
            tgt : (batch_size, target_sequence_length, input_size)
            -> TransformerのDecoder側の入力
        Outputs:
            output : (batch_size, target_sequence_length, hidden_size)
            -> Transformerの最終出力
        """
        
        src = src.permute(1, 0, 2)
        """ src : (source_sequence_length, batch_size, hidden_size) """        

        tgt = self.tgt_fc(tgt)
        """ tgt : (batch_size, target_sequence_length, input_size) """
        tgt_pe = self.tgt_pe(tgt).permute(1, 0, 2)
        """ tgt : (target_sequence_length, batch_size, hidden_size) """
        output = self.decoder(tgt_pe, src)
        """ output : (target_sequence_length, batch_size, hidden_size) """
        output = output.permute(1, 0, 2)
        """ output : (batch_size, target_sequence_length, hidden_size) """
        return output