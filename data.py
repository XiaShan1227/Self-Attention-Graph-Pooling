#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/18 15:52
"""

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset


def load_dataset(args):
    # 每张图包括x；edge_index；y，图分类
    dataset = TUDataset(root='Data/TUDataset', name=args.dataset)

    num_train = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_train + num_val)

    # 8:1:1划分数据集
    train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset.num_features, dataset.num_classes


if __name__ == '__main__':
    dataset = TUDataset(root='Data/TUDataset', name='DD')
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    first_graph_data = dataset[0]  # 获取第一个图对象

    print()
    print(first_graph_data)
    print('=' * 46)

    # 获取第一张图的统计信息
    print(f'Number of nodes: {first_graph_data.num_nodes}')
    print(f'Number of edges: {first_graph_data.num_edges}')
    print(f'Average node degree: {first_graph_data.num_edges / first_graph_data.num_nodes:.2f}')
    print(f'Has isolated nodes: {first_graph_data.has_isolated_nodes()}')
    print(f'Has self-loops: {first_graph_data.has_self_loops()}')
    print(f'Is undirected: {first_graph_data.is_undirected()}')
