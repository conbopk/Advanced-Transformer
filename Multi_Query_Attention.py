import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.wq = nn.Linear(hid_dim, hid_dim)
        self.wk = nn.Linear(hid_dim, self.head_dim)
        self.wv = nn.Linear(hid_dim, self.head_dim)

        self.dense = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def split_heads(self, x, batch_size, num_heads, head_dim):
        x = x.view(batch_size, -1, num_heads, head_dim)

        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = self.split_heads(q, batch_size, self.n_heads, self.head_dim)

        k = self.split_heads(k, batch_size, 1, self.head_dim).permute(0, 1, 3, 2)
        v = self.split_heads(v, batch_size, 1, self.head_dim)

        energy = torch.matmul(q, k) / self.scale  # [batch_size, n_heads, seq_len, key_len]

        if mask is not None:
            energy = energy.masked_fill_(mask == 0, -1e10)

        attention_weights = F.softmax(energy, dim=-1)

        scaled_attention = torch.matmul(self.dropout(attention_weights), v)  # [batch_size, n_heads, seq_len, head_dim]

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.hid_dim)  # [batch_size, seq_len, hid_dim]

        output = self.dense(original_size_attention)

        return output, attention_weights


hid_dim = 512
n_heads = 8
dropout = 0.1
seq_len = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mqa_layer = MultiQueryAttentionLayer(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout, device=device).to(device)

batch_size = 2
query = torch.randn(batch_size, seq_len, hid_dim).to(device)
key = torch.randn(batch_size, seq_len, hid_dim).to(device)
value = torch.randn(batch_size, seq_len, hid_dim).to(device)

output, attention_weights = mqa_layer(query, key, value)

print("Output shape: ", output.shape)
print('Attention weights shape:', attention_weights.shape)
