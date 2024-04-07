【ICML-2019 SAGPooling】[Self-Attention Graph Pooling](https://arxiv.org/pdf/1904.08082.pdf)
![image](https://github.com/XiaShan1227/Self-Attention-Graph-Pooling/assets/67092235/7965f5ca-ea1f-4bcf-8ff2-015399a5ab28)
![image](https://github.com/XiaShan1227/Self-Attention-Graph-Pooling/assets/67092235/67496a3a-5860-4fc2-a942-21861b0e9f8f)

1.实验参数
| **Parameter** | **Value** |
|:-------------| :------------:|
| Batch size | 128 |
| Dataset | 可选: DD、MUTAG、NCI1、NCI109、PROTEINS, etc |
| Dropout ratio | 0.5 |
| Epochs | 10000 |
| Exp name | 自命名: DD_Glo、MUTAG_Hie, etc |
| Gpu index | 0 |
| Hid | 128 |
| Lr | 0.0005 |
| Model | 可选: SAGPooling_Global、SAGPooling_Hierarchical |
| Patience | 40 |
| Pooling ratio | 0.5 |
| Seed | 16 |
| Test batch size | 1 |
| Weight decay | 0.0001 |

2.运行程序 </br>
模型：SAGPooling_Global </br>
数据集：DD
```python
python main.py --exp_name=DD_Glo --dataset=DD --model=SAGPooling_Global
```

模型：SAGPooling_Hierarchical </br>
数据集：PROTEINS
```python
python main.py --exp_name=PROTEINS_Hie --dataset=PROTEINS --model=SAGPooling_Hierarchical
```

3.实验结果（8:1:1划分数据集，只做了一次实验的准确率，保留两位小数）
| **DD** | **MUTAG** | **NCI1** | **Value** | **Value** |
|:-------------|:------------:|:------------:|:------------:|:------------:|
| SAGPooling_Global       |  73.11  |  80.00  |  69.10  |  74.40  |  73.21  |
| SAGPooling_Hierarchical |  67.23  |  70.00  |  66.18  |  70.77  |  69.64  |
