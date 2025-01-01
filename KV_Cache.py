import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        #x shape: [batch_size, seq_len, d_model]
        x = x.view(batch_size, -1, self.num_heads, self.depth)  #[batch_size, seq_len, num_heads, depth]
        return x.permute(0, 2, 1, 3)    #[batch_size, num_heads, seq_len, depth]

    def forward(self, x, prev_K=None, prev_V=None, tag=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        q = self.W_q(x)     #[batch_size, seq_len, d_model]
        k = self.W_k(x)     #[batch_size, seq_len, d_model]
        v = self.W_v(x)     #[batch_size, seq_len, d_model]

        q = self.split_heads(q, batch_size)     #[batch_size, num_heads, seq_len, depth]
        k = self.split_heads(k, batch_size)     #[batch_size, num_heads, seq_len, depth]
        v = self.split_heads(v, batch_size)     #[batch_size, num_heads, seq_len, depth]

        if prev_K is not None and prev_V is not None:
            print('before ===>', 'x', x.shape, "q", q.shape, "k", k.shape, "v", v.shape, "prev_K", prev_K.shape, "prev_V", prev_V.shape)

            # Ensure that prev_K and prev_V have the same number of dimensions as k and v
            if prev_K.dim() == 3:
                prev_K= prev_K.view(batch_size, self.num_heads, -1, self.depth)
            if prev_V.dim() == 3:
                prev_V = prev_V.view(batch_size, self.num_heads, -1, self.depth)

            # Concatenate along the sequence length dimension
            k = torch.cat([prev_K, k], dim=2)   #[batch_size, num_heads, m + seq_len, depth]
            v = torch.cat([prev_V, v], dim=2)   #[batch_size, num_heads, m + seq_len, depth]

            print('After ===>', 'x', x.shape, "q", q.shape, "k", k.shape, "v", v.shape, "prev_K", prev_K.shape, "prev_V", prev_V.shape)

        # Scaled dot-product attention
        logits = torch.matmul(q, k.transpose(-2,-1))    #[batch_size, num_heads, seq_len, m + seq_len]
        logits /= math.sqrt(self.depth)
        weights = torch.softmax(logits, dim=-1)
        attention = torch.matmul(weights, v)            #[batch_size, num_heads, seq_len, depth]

        # Concatenate heads and project back to d_model
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(batch_size, seq_len, self.d_model)
        output = self.W_o(attention)                    #[batch_size, seq_len, d_model]

        return output, k, v


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     #[max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  #[d_model/2]
        self.pe[:, 0::2] = torch.sin(position * div_term)   #[max_len, d_model]
        self.pe[:, 1::2] = torch.cos(position * div_term)   #[max_len, d_model]
        self.pe = self.pe.unsqueeze(0)  #[1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, prev_K=None, prev_V=None):
        # Self-attention (masked for autoregressive behavior)
        print("-------> Caching -------->")
        _x, k, v = self.self_attn(x, prev_K, prev_V)
        print("<------- End Caching <--------")

        x = x + self.dropout(_x)
        x = self.norm1(x)

        # Encoder-Decoder attention
        _x, _, _, = self.enc_dec_attn(x, enc_out, enc_out)
        x = x + self.dropout(x)
        x = self.norm2(x)

        # Feed-forward network
        _x = self.ffn(x)
        x = x + self.dropout(_x)
        x = self.norm3(x)

        return x, k, v

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers,d_ff, vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, enc_out, prev_K_list=None, prev_V_list=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        if prev_K_list is None or prev_V_list is None:
            prev_K_list = [torch.empty(0) for _ in range(len(self.layers))]
            prev_V_list = [torch.empty(0) for _ in range(len(self.layers))]

        new_K_list = []
        new_V_list = []

        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, enc_out, prev_K_list[i], prev_V_list[i])
            print('===>', x.shape, k.shape, v.shape)
            if prev_K_list[i].numel() > 0 and prev_V_list[i].numel() > 0:
                print('*', prev_K_list[i].shape, prev_V_list[i].shape)

            new_K_list.append(k)
            new_V_list.append(v)

        logits = self.linear(x)

        return logits, new_K_list, new_V_list

d_model = 512
num_heads = 8
num_layers = 1
d_ff = 2048
vocab_size = 10000
max_seq_len = 100

decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff, vocab_size, max_seq_len)

# Assume `decoder` is an instance of `TransformerDecoder`
# `enc_output` is the encoder's output that the decoder will attend to
# `start_token` is the initial token to start the decoding process

start_token = 1
end_token = 2
enc_output = torch.randn(1, max_seq_len, d_model, dtype=torch.float)

generated_sequence = [start_token]
prev_K_list, prev_V_list = None, None

for i in range(max_seq_len):
    print('----> Decoding ...')
    current_input = torch.tensor([generated_sequence[-1]]).unsqueeze(0)
    print(current_input.shape)

    # Forward pass through the decoder
    logits, new_K_list, new_V_list = decoder(current_input, enc_output, prev_K_list, prev_V_list)

    next_token = logits.argmax(dim=-1).item()

    generated_sequence.append(next_token)

    prev_K_list, prev_V_list = new_K_list, new_V_list

    if next_token == end_token:
        break

    print('---> End Decoding ...')








