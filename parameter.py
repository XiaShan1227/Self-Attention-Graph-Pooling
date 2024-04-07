#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/18 16:03
"""

import argparse
from texttable import Texttable


def parse_args():
    parser = argparse.ArgumentParser()  # 参数解析器对象

    parser.add_argument('--seed', type=int, default=16, help='Random seed of the experiment')  # 随机种子
    parser.add_argument('--exp_name', type=str, default='Exp', help='Name of the experiment')
    parser.add_argument('--dataset', type=str, help='Dataset to use: [DD, PROTEINS, NCI1, etc]', required=True)
    parser.add_argument('--model', type=str, default='SAGPooling_Hierarchical', choices=['SAGPooling_Global', 'SAGPooling_Hierarchical'], help='Model to use') # 使用全局池化或者分层池化模型
    parser.add_argument('--batch_size', type=int, default=128, help='Size of the training and validation batch')  # 训练、验证集batch_size
    parser.add_argument('--test_batch_size', type=int, default=1, help='Size of the testing batch')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument('--epochs', type=int, default=10000, help='Maximum number of epochs')
    parser.add_argument('--hid', type=int, default=128, help='Size of the hidden layer')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='Graph pooling ratio')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay of Adam')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=40, help='Patience for early stopping')

    args = parser.parse_args()  # 解析命令行参数

    return args


class IOStream():
    """训练日志文件"""
    def __init__(self, path):
        self.file = open(path, 'a') # 附加模式：用于在文件末尾添加内容，如果文件不存在则创建新文件

    def cprint(self, text):
        print(text)
        self.file.write(text + '\n')
        self.file.flush() # 确保将写入的内容刷新到文件中，以防止数据在缓冲中滞留

    def close(self):
        self.file.close()


def table_printer(args):
    """绘制参数表格"""
    args = vars(args) # 转成字典类型
    keys = sorted(args.keys()) # 按照字母顺序进行排序
    table = Texttable()
    table.set_cols_dtype(['t', 't']) # 列的类型都为文本(str)
    rows = [["Parameter", "Value"]] # 设置表头
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])]) # 下划线替换成空格，首字母大写
    table.add_rows(rows)
    return table.draw()
