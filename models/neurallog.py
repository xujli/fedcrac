import torch.nn as nn

import torch
# embed_dim = 768  # Embedding size for each token
# num_heads = 12  # Number of attention heads
# ff_dim = 2048  # Hidden layer size in feed forward network inside transformer
# max_len = 20

def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / torch.tensor(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(torch.unsqueeze(torch.arange(position), 1),
                            torch.unsqueeze(torch.arange(d_model), 0),
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.unsqueeze(angle_rads, 0)

    return pos_encoding.float()

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim, device):
        super(PositionEmbedding, self).__init__()
        self.pos_encoding = positional_encoding(max_len,
                                                embed_dim).to(device)

    def forward(self, x):
        seq_len = x.shape[1]
        x += self.pos_encoding[:, :seq_len, :]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.ffn = nn.Sequential(
            *[nn.Linear(embed_dim, ff_dim), nn.Linear(ff_dim, embed_dim)]
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output[0])
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class NeuralLog(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, ff_dim=1024, num_heads=12, dropout=0.1, device='cuda', KD=False):
        super(NeuralLog, self).__init__()
        self.embedding_layer = PositionEmbedding(1024, embed_dim, device)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(embed_dim, 32)
        self.final = nn.Linear(32, num_classes)
        self.KD = KD

    def forward(self, inputs, device=None):
        if type(inputs) == list:
            inputs = inputs[0]
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = torch.transpose(x, 1, 2)
        x = self.pooling(x)
        x_pool = self.dropout(torch.flatten(x, 1))
        x = self.linear(x_pool)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final(x)
        return (x_pool, x) if self.KD else x 
    
class Embedding(nn.Module):
    def __init__(self, max_len, embed_dim, device):
        super(Embedding, self).__init__()
        self.pos_encoding = positional_encoding(max_len,
                                                embed_dim).to(device)

    def forward(self, x):
        seq_len = x.shape[1]
        x =x+  self.pos_encoding[:, :seq_len, :].clone()
        return x

class TransformerEncoding(nn.Module):
    def __init__(self, num_lay, embed_dim, num_heads, ff_dim, rate=0.1) -> None:
        super().__init__()
        self.model = nn.Sequential()
        for i in range(num_lay):
            self.model.add_module('i',TransformerBlock(embed_dim, num_heads, ff_dim, rate))
    def forward(self,x):
        return self.model(x)
    
class Head(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, dropout=0.1, KD=True) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(embed_dim, 32)
        self.final = nn.Linear(32, num_classes)
        self.KD = KD
    def forward(self,x):
        x = torch.transpose(x, 1, 2)
        x = self.pooling(x)
        x = self.dropout(torch.flatten(x, 1))
        x = self.linear(x)
        x1 = self.relu(x)
        x = self.dropout(x)
        x = self.final(x)
        if self.KD:
            return x1, x
        else:
            return x


if __name__ == '__main__':
    model = Head(embed_dim=768, num_classes=1, dropout=0.1)
    inputs = torch.zeros((1, 20, 768))
    output = model(inputs)
    print(output.shape)