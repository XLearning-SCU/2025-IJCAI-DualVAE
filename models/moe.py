import torch
import torch.nn as nn
import torch.nn.functional as F



class Moe(nn.Module):
    def __init__(self, views, input_dim, output_dim):
        super(Moe, self).__init__()
        self.num_experts = views
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = (input_dim+output_dim)*2//3

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            ) for _ in range(self.num_experts)
        ])
        self.gates = nn.Linear(input_dim*views, self.num_experts)

    def forward(self, Xs):
        gate_input = torch.cat(Xs, dim=-1)
        gate_score = F.softmax(self.gates(gate_input), dim=-1) #(b,m)
        expers_output = [self.experts[i](Xs[i]) for i in range(self.num_experts)]
        expers_output = torch.stack(expers_output, dim=1)  #(b,m,2c)
        output = torch.bmm(gate_score.unsqueeze(1), expers_output).squeeze(1)

        return output
