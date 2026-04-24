import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        # Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Gate parameters (Part 1.2)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.gate_scores, 0.5) 

    def forward(self, x):
        # Part 1.3: Apply Sigmoid to get gates
        gates = torch.sigmoid(self.gate_scores)
        # Element-wise multiplication for pruning
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity_loss(self):
        """Part 2.2: L1 norm of all gates across the network."""
        l1_sum = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                l1_sum += torch.sum(torch.sigmoid(m.gate_scores))
        return l1_sum

    def get_sparsity_level(self, threshold=1e-2):
        """Part 3.1: Percentage of weights pruned."""
        total, pruned = 0, 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                total += gates.numel()
                pruned += (gates < threshold).sum().item()
        return (pruned / total) * 100

    def get_all_gates(self):
        """Helper to collect all gate values for the histogram."""
        all_gates = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                all_gates.append(torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten())
        return np.concatenate(all_gates)