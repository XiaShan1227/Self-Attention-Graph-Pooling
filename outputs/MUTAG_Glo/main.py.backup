#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/18 17:16
"""

import os, torch, random
import torch.optim as optim
import torch.nn.functional as F

from data import load_dataset
from model import SAGPooling_Global, SAGPooling_Hierarchical
from parameter import parse_args, IOStream, table_printer


def train(args, IO, train_loader, val_loader, num_features, num_classes):
    # 使用GPU or CPU
    device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index))
    if args.gpu_index < 0:
        IO.cprint('Using CPU')
    else:
        IO.cprint('Using GPU: {}'.format(args.gpu_index))
        torch.cuda.manual_seed(args.seed)  # 设置PyTorch GPU随机种子

    # 加载模型及参数量统计
    if args.model == 'SAGPooling_Global': # 全局池化模型
        model = SAGPooling_Global(args, num_features, num_classes).to(device)
        IO.cprint(str(model))
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        IO.cprint('Model Parameter: {}'.format(total_params))
    elif args.model == 'SAGPooling_Hierarchical':  # 分层池化模型
        model = SAGPooling_Hierarchical(args, num_features, num_classes).to(device)
        IO.cprint(str(model))
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        IO.cprint('Model Parameter: {}'.format(total_params))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    IO.cprint('Using Adam')

    min_loss = 1e10
    patience = 0

    for epoch in range(args.epochs):
        #################
        ###   Train   ###
        #################
        model.train()  # 训练模式
        train_loss_epoch = [] # 记录每个epoch的训练损失

        for i, data in enumerate(train_loader):
            data = data.to(device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_epoch.append(round(float(loss.cpu()), 6))

        # 将 loss 列表中的所有元素转换为字符串，并以逗号分隔拼接
        loss_str = ', '.join(['{:.4f}'.format(i) for i in train_loss_epoch])

        IO.cprint('')
        IO.cprint('Epoch #{:03d}, Train_Loss: [{}]'.format(epoch, loss_str))

        ################
        ##  Validate  ##
        ################
        model.eval()  # 评估模式
        correct = 0.
        loss = 0.

        for i, data in enumerate(val_loader):
            data = data.to(device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            loss += F.nll_loss(out, data.y, reduction='sum').item()

        val_acc = correct / len(val_loader.dataset)
        val_loss = loss / len(val_loader.dataset)

        IO.cprint('Val_Loss: {:.6f}, Val_Acc: {:.6f}'.format(val_loss, val_acc))

        if val_loss < min_loss:
            torch.save(model, 'outputs/%s/model.pth' % args.exp_name)
            IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break


def test(args, IO, test_loader):
    """测试模型"""
    device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index))

    # 输出内容保存在之前的训练日志里
    IO.cprint('')
    IO.cprint('********** TEST START **********')
    IO.cprint('Reload Best Model')
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

    model = torch.load('outputs/%s/model.pth' % args.exp_name).to(device)
    model = model.eval()  # 创建一个新的评估模式的模型对象，不覆盖原模型

    ################
    ###   Test   ###
    ################
    correct = 0.

    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    test_acc = correct / len(test_loader.dataset)
    IO.cprint('TEST :: Test_Acc: {:.6f}'.format(test_acc))


def exp_init():
    """实验初始化"""
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.mkdir('outputs/' + args.exp_name)

    # 跟踪执行脚本，windows下使用copy命令，且使用双引号
    os.system(f"copy main.py outputs\\{args.exp_name}\\main.py.backup")
    os.system(f"copy data.py outputs\\{args.exp_name}\\data.py.backup")
    os.system(f"copy model.py outputs\\{args.exp_name}\\model.py.backup")
    os.system(f"copy parameter.py outputs\\{args.exp_name}\\parameter.py.backup")
    # os.system('cp main.py outputs' + '/' + args.exp_name + '/' + 'main.py.backup')
    # os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')
    # os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    # os.system('cp parameter.py outputs' + '/' + args.exp_name + '/' + 'parameter.py.backup')


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    exp_init()

    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化

    train_loader, val_loader, test_loader, num_features, num_classes = load_dataset(args)

    train(args, IO, train_loader, val_loader, num_features, num_classes)
    test(args, IO, test_loader)