import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, input_vector_size = 16, hidden_size = 16, input_context_size = None):
        super(SelfAttention, self).__init__()
        if input_context_size is not None:
            self.W_a = torch.randn(input_context_size, hidden_size, requires_grad=True)
        else:
            self.W_a = None
        self.U_a = torch.randn(input_vector_size, hidden_size, requires_grad=True)
        self.v_a = torch.randn(1, hidden_size, requires_grad=True)

    def forward(self, vectors, predicate = None):
        if self.W_a is None or predicate is None:
            pre_activation_energies = F.tanh(vectors @ self.U_a)
        else:
            pre_activation_energies = F.tanh(predicate @ self.W_a + vectors @ self.U_a)
        activation_energies = torch.einsum("tbd,ad->tb", (pre_activation_energies.clone(), self.v_a.clone()))

        return torch.t(activation_energies)

    def combine(self, vectors, predicate = None):
        activation_energies = torch.t(self(vectors, predicate))
        activations = F.softmax(activation_energies, dim=0)
        context = torch.einsum("tb,tbd->bd", (activations.clone(), vectors.clone()))
        return context
