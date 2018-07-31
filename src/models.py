import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import SelfAttention

class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers, feature_size, batch_size):
        super(RNN, self).__init__()
        # HyperParameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.batch_size = batch_size

        # Parameters
        #self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
        self.cells = nn.ModuleList([nn.GRUCell(input_size=self.feature_size, hidden_size=hidden_size)] +
                                   [nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])


        self.dropout1 = nn.Dropout(.3)
        self.fc1 = nn.Linear(hidden_size,32)
        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(.3)
        self.fc2 = nn.Linear(32,1)

    def forward(self, batch):
        # (sequence, batch, feature)
        #embedded_input = self.embedding(batch)
        h_s = [torch.zeros((self.batch_size, self.hidden_size)) for i in range(self.num_layers)]
        out_vecs = []
        for token in batch.split(1):
            input_vec = token.squeeze().view(1,self.feature_size)
            # get all outputs (go up)
            new_h_s = []
            for h, cell in zip(h_s, self.cells):
                input_vec = cell(input_vec, h)
                new_h_s.append(input_vec)

            h_s = new_h_s

            do1 = self.dropout1(new_h_s[-1].squeeze())
            fc1 = self.fc1(do1)
            fc1 = self.relu1(fc1)

            do2 = self.dropout2(fc1)
            fc2 = self.fc2(do2)

            out_vecs.append(fc2)
        return out_vecs

class FBRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, feature_size, attention_hidden_size,embedding_dim=None, vocab_size=None):
        super(FBRNN, self).__init__()
        # HyperParameters
        #self.embedding_dim = embedding_dim
        #self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_hidden_size = attention_hidden_size
        self.batch_size = batch_size
        self.feature_size = feature_size

        # Parameters
        #self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
        self.cells = nn.ModuleList([nn.GRUCell(input_size=self.feature_size, hidden_size=hidden_size)] +
                                   [nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])
        self.hidden_attentions = nn.ModuleList([SelfAttention(input_vector_size=hidden_size, hidden_size=attention_hidden_size) for i in range(num_layers)])

        self.dropout1 = nn.Dropout(.3)
        self.fc1 = nn.Linear(16,32)
        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(.3)
        self.fc2 = nn.Linear(32,1)

    def forward(self, batch):
        #embedded_input = self.embedding(batch)
        batch_size = batch.shape[1]

        h_s = [torch.zeros((self.batch_size, self.hidden_size)) for i in range(self.num_layers)]

        out_vecs = []
        for token in batch.split(1):
            input_vec = token.squeeze().view(1,self.feature_size)
            # get all outputs (go up)
            new_h_s = []
            for h, cell in zip(h_s, self.cells):
                input_vec = cell(input_vec, h)
                new_h_s.append(input_vec)

            h_s = []
            # compute new hidden states using attention (go right)
            for i, att in enumerate(self.hidden_attentions):
                h_s.append(att.combine(torch.stack(new_h_s[i:])))

            do1 = self.dropout1(h_s[-1].squeeze())
            fc1 = self.fc1(do1)
            fc1 = self.relu1(fc1)

            do2 = self.dropout2(fc1)
            fc2 = self.fc2(do2)

            out_vecs.append(fc2)
        return out_vecs


class FBRNN_encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, attention_hidden_size,embedding_dim=None, vocab_size=None):
        super(FBRNN_encoder, self).__init__()
        # HyperParameters
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Parameters
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.cells = nn.ModuleList([nn.GRUCell(input_size=embedding_dim, hidden_size=hidden_size)] +
                                   [nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])
        self.hidden_attentions = nn.ModuleList([SelfAttention(input_vector_size=hidden_size, hidden_size=attention_hidden_size) for i in range(num_layers)])

    def init_hidden(self,device):
        self.h_s = [torch.zeros((1,self.hidden_size),device=device) for i in range(self.num_layers)]

    def forward(self, batch):
        embedded_input = self.embedding(batch)
        print('Encoding',batch.shape)
        batch_size = batch.shape[1]

        out_vecs = []
        for token in embedded_input.split(1):
            input_vec = token.squeeze().view(1,self.embedding_dim)
            # get all outputs (go up)
            new_h_s = []
            for h, cell in zip(self.h_s, self.cells):
                input_vec = cell(input_vec, h)
                new_h_s.append(input_vec)

            self.h_s = []
            # compute new hidden states using attention (go right)
            for i, att in enumerate(self.hidden_attentions):
                self.h_s.append(att.combine(torch.stack(new_h_s[i:])))

            out_vecs.append(self.h_s[-1].squeeze())
        return torch.stack(out_vecs)


class FBRNN_decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, attention_hidden_size,embedding_dim=None, vocab_size=None):
        super(FBRNN_decoder, self).__init__()
        # HyperParameters
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Parameters
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.cells = nn.ModuleList([nn.GRUCell(input_size=2*embedding_dim, hidden_size=hidden_size)] +
                                   [nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for i in range(num_layers-1)])
        self.hidden_attentions = nn.ModuleList([SelfAttention(input_vector_size=hidden_size, hidden_size=attention_hidden_size) for i in range(num_layers)])

        self.attn = SelfAttention(input_vector_size=hidden_size, hidden_size=attention_hidden_size, input_context_size=hidden_size)
        self.dropout1 = nn.Dropout(.3)
        self.fc1 = nn.Linear(hidden_size,32)
        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(.3)
        self.fc2 = nn.Linear(32,vocab_size)

    def init_hidden(self,device):
        self.h_s = [torch.zeros((1,self.hidden_size),dtype=torch.float64,device=device) for i in range(self.num_layers)]

    def forward(self, decoder_input, encoder_outputs):
        # decoder_input: correct token
        embedded_decoder_input = self.embedding(decoder_input)
        encoder_outputs = encoder_outputs.unsqueeze(1)
        out_vecs = []
        for token in embedded_decoder_input.split(1):
            attn = self.attn.combine(encoder_outputs,self.h_s[0])
            input_vec = torch.cat([attn.unsqueeze(0),token]).view(-1,2*self.hidden_size)

            # get all outputs (go up)
            new_h_s = []
            for h, cell in zip(self.h_s, self.cells):
                input_vec = cell(input_vec, h)
                new_h_s.append(input_vec)

            self.h_s = []
            # compute new hidden states using attention (go right)
            for i, att in enumerate(self.hidden_attentions):
                self.h_s.append(att.combine(torch.stack(new_h_s[i:])))

            do1 = self.dropout1(self.h_s[-1].squeeze())
            fc1 = self.fc1(do1)
            fc1 = self.relu1(fc1)

            do2 = self.dropout2(fc1)
            fc2 = self.fc2(do2)
            out = F.softmax(fc2,0)

            out_vecs.append(out)
        return torch.stack(out_vecs)
