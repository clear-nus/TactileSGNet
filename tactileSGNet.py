import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, ChebConv, TAGConv
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x, global_mean_pool)

import numpy as np
from to_graph import *

thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 
"""
This code about LIF neuron update is based on the work by Wu et al., which is available 
https://github.com/yjwu17/BP-for-SpikingNN
"""

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update, for GCN
def mem_update_conv(ops, x, edge_idxs, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x, edge_idxs)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem)
    return mem, spike

# cnn_layer(in_channels, out_channels)
cfg_cnn = [(2, 64),
    #       (64, 128)
           ] # 
# kernel size
cfg_s = [39, 39] # 

# fc layer
cfg_fc = [128, 256]
gamma = 0.2 # dropout coefficient
print('Model: TactileSGNet')
print(cfg_cnn)
print(cfg_fc)

class TactileSGNet(nn.Module):
    def __init__(self, num_classes=36, k=0, device="cuda:0"):
        super(TactileSGNet, self).__init__()
        in_planes, out_planes = cfg_cnn[0]
        self.conv1 = TAGConv(in_planes, out_planes, K=3)

        self.fc1 = nn.Linear(cfg_s[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], num_classes)
        self.num_classes = num_classes
        self.graph = TactileGraph(k)
        self.device = device

    def forward(self, input, training = True):
        data = input.to(self.device)
        sizes = data.size()
        time_window = sizes[-1]

        c1_mem = c1_spike = c1_mem_noise = torch.zeros(cfg_s[0], cfg_cnn[0][1], device=self.device)
        h1_mem = h1_spike = h1_sumspike = h1_mem_noise = torch.zeros(cfg_fc[0], device=self.device)
        h2_mem = h2_spike = h2_sumspike = h2_mem_noise = torch.zeros(cfg_fc[1], device=self.device)
        h3_mem = h3_spike = h3_sumspike = h3_mem_noise = torch.zeros(self.num_classes, device=self.device)
        
        inputs = data.split(1, dim=len(sizes)-1)
        for step in range(time_window): # simulation time steps
            x = inputs[step].squeeze()
            x = x.to(self.device)
            graph_data = self.graph(x)
            x = graph_data.x.to(self.device) 
            edge_idxs = graph_data.edge_index.to(self.device)

            c1_mem, c1_spike = mem_update_conv(self.conv1, x, edge_idxs, c1_mem, c1_spike)
            x = c1_spike
            x = x.view(-1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike

        outputs = h3_sumspike / time_window
        return outputs


