import torch
import torch.nn as nn
from attention import SelfAttention

class FBRNN(nn.Module):

    def __init__(self, embedding_dim, vocab_size, hidden_size, num_layers, attention_hidden_size):
        super(FBRNN, self).__init__()
        # HyperParameters
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_hidden_size = attention_hidden_size

        # Parameters
        self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
        self.cells = nn.ModuleList([nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)] +
                                   [nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])
        self.hidden_attentions = nn.ModuleList([SelfAttention(input_vector_size=hidden_size, hidden_size=attention_hidden_size) for i in range(num_layers)])

    def forward(self, batch):
        embedded_input = self.embedding(batch)
        batch_size = batch.shape[1]
        h_s = [torch.zeros((batch_size, self.hidden_size)) for i in range(self.num_layers)]

        out_vecs = []
        for token in embedded_input.split(1):
            input_vec = token.squeeze()
            # get all outputs (go up)
            new_h_s = []
            for h, cell in zip(h_s, self.cells):
                input_vec = cell(input_vec, h)
                new_h_s.append(input_vec)

            h_s = []
            # compute new hidden states using attention (go right)
            for i, att in enumerate(self.hidden_attentions):
                h_s.append(att.combine(torch.stack(new_h_s[i:])))
            out_vecs.append(h_s[-1])
        return out_vecs
