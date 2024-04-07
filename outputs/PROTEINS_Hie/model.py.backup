#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/18 17:36
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class SAGPooling_Global(nn.Module):
    def __init__(self, args, num_features, num_classes):
        super(SAGPooling_Global, self).__init__()
        self.hid = args.hid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(num_features, self.hid)
        self.conv2 = GCNConv(self.hid, self.hid)
        self.conv3 = GCNConv(self.hid, self.hid)

        self.pool = SAGPooling(self.hid * 3, ratio=self.pooling_ratio, GNN=GCNConv, min_score=None, nonlinearity=torch.tanh)

        self.lin1 = nn.Linear(self.hid * 6, self.hid)
        self.lin2 = nn.Linear(self.hid, self.hid // 2)

        self.classifier = nn.Linear(self.hid // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, edge_index)) # (num_nodes, num_features=89) ——> (num_nodes, num_features=hid=128)
        x2 = F.relu(self.conv2(x1, edge_index)) # (num_nodes, num_features=hid=128) ——> (num_nodes, num_features=hid=128)
        x3 = F.relu(self.conv3(x2, edge_index)) # (num_nodes, num_features=hid=128) ——> (num_nodes, num_features=hid=128)

        x = torch.cat([x1, x2, x3], dim=1)  # (num_nodes, num_features=128) ——> (batch_size, num_features=128*3=384)

        x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch)  # (num_nodes, num_features=hid*3=384) ——> (num_nodes1 = num_nodes * pooling_ratio, num_features=hid*3=384)

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # (num_nodes1, num_features=384) ——> (batch_size, num_features=384*2=768)

        x = F.relu(self.lin1(x)) # (batch_size, num_features=768) ——> (batch_size, num_features=128)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x)) # (batch_size, num_features=128) ——> (batch_size, num_features=64)

        x = self.classifier(x)  # (batch_size, 64) ——> (batch_size, datasets_number_categories)

        x = F.log_softmax(x, dim=-1)

        return x


class SAGPooling_Hierarchical(nn.Module):
    def __init__(self, args, num_features, num_classes):
        super(SAGPooling_Hierarchical, self).__init__()
        self.hid = args.hid
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(num_features, self.hid)
        self.conv2 = GCNConv(self.hid, self.hid)
        self.conv3 = GCNConv(self.hid, self.hid)

        self.pool = SAGPooling(self.hid, ratio=self.pooling_ratio, GNN=GCNConv, min_score=None, nonlinearity=torch.tanh)

        self.lin1 = nn.Linear(self.hid * 2, self.hid)
        self.lin2 = nn.Linear(self.hid, self.hid // 2)

        self.classifier = nn.Linear(self.hid // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index)) # (num_nodes, num_features=89) ——> (num_nodes, num_features=hid=128)
        x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch) # (num_nodes, num_features=128) ——> (num_nodes1 = num_nodes * pooling_ratio, num_features=hid=128)

        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # (num_nodes1, num_features=128) ——> (batch_size, num_features=128*2=256)

        x = F.relu(self.conv2(x, edge_index)) # (num_nodes1, num_features=hid=128) ——> (num_nodes1, num_features=hid=128)
        x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch) # (num_nodes1, num_features=hid=128) ——> (num_nodes2 = num_nodes1 * pooling_ratio, num_features=hid=128)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1) # (num_nodes2, num_features=128) ——> (batch_size, num_features=128*2=256)

        x = F.relu(self.conv3(x, edge_index)) # (num_nodes2, num_features=hid=128) ——> (num_nodes2, num_features=hid=128)
        x, edge_index, _, batch, perm, score = self.pool(x, edge_index, None, batch)  # (num_nodes2, num_features=hid=128) ——> (num_nodes3 = num_nodes2 * pooling_ratio, num_features=hid=128)

        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # (num_nodes3, num_features=128) ——> (batch_size, num_features=128*2=256)

        x = x1 + x2 + x3 # (batch_size, num_features=256)

        x = F.relu(self.lin1(x)) # (batch_size, num_features=256) ——> (batch_size, num_features=128)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x)) # (batch_size, num_features=128) ——> (batch_size, num_features=64)

        x = self.classifier(x)  # (batch_size, 64) ——> (batch_size, datasets_number_categories)

        x = F.log_softmax(x, dim=-1)

        return x
