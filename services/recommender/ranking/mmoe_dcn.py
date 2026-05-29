import torch
import torch.nn as nn

class CrossNetworkV2(nn.Module):
    def __init__(self, input_dim: int, rank: int = 16):
        super().__init__()
        # Low-Rank parameterization matrix reduction optimization: W ~ U * V^T
        self.U = nn.Parameter(torch.randn(input_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim, 1))
        
    def forward(self, x0, x_l):
        # x_l is [batch, input_dim] -> transpose for low rank scaling
        x_col = x_l.unsqueeze(-1)
        # Compute V * x_l
        proj = torch.matmul(self.V, x_col)
        # Compute U * V * x_l
        prod = torch.matmul(self.U, proj) + self.bias
        # Hadamard outer element multiplication
        x_next = x0 * prod.squeeze(-1) + x_l
        return x_next

class MMoEDCNRanker(nn.Module):
    def __init__(self, input_dim: int, num_experts: int = 4, num_tasks: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        
        # 1. Low Rank Cross Network Crossing Layer
        self.cross_net = CrossNetworkV2(input_dim, rank=8)
        
        # 2. Shared Multi-Gate Experts Networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, embedding_dim)
            ) for _ in range(num_experts)
        ])
        
        # 3. Softmax Gating routing distributions (Task Specific)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
        
        # 4. Multi-Task towers (Tower 0: CTR Prediction, Tower 1: CVR Prediction)
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, x):
        # Apply Cross Networks Feature Crossings
        crossed_x = self.cross_net(x, x)
        
        # Collect expert transformations
        expert_outputs = [expert(crossed_x).unsqueeze(1) for expert in self.experts]
        expert_outputs = torch.cat(expert_outputs, dim=1) # Shape: [batch, num_experts, embedding_dim]
        
        task_outputs = []
        for i in range(self.num_tasks):
            # Compute expert routing gate values
            gate_weights = self.gates[i](crossed_x).unsqueeze(-1) # Shape: [batch, num_experts, 1]
            # Weighted sum over experts
            expert_blend = (expert_outputs * gate_weights).sum(dim=1)
            
            # Route blended vectors through task specific head towers
            task_outputs.append(self.towers[i](expert_blend))
            
        return task_outputs # Returns list [ctr_probs, cvr_probs]
