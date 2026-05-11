# CIFAR-10 STDP SNN

本项目使用器件实验拟合得到的 STDP 参数，在 CIFAR-10 数据集上训练脉冲神经网络。
训练时必须读取当前目录下的 `stdp_params.npz`。这是本项目的硬性要求；
如果文件不存在，程序会直接报错，不会自动改用其他参数文件。

当前稳定版本使用 `conv-stdp` 模型，在 CIFAR-10 测试集上约为 `66%` 准确率：

```text
checkpoint=checkpoints/conv_stdp/best.pt
checkpoint_epoch=118
saved_test_acc=0.6646
```

## 项目结构

```text
data/           CIFAR-10 数据加载与 Poisson 脉冲编码
model/          STDP 突触层、LIF 模型、卷积 STDP-SNN 模型
utils/          随机种子、设备选择、STDP 参数读取、checkpoint 工具
tools/          器件 STDP 参数提取脚本
checkpoints/    训练输出目录

train.py        训练入口
eval.py         测试集评估入口
infer.py        checkpoint 推理演示入口
stdp_params.npz 器件拟合得到的 STDP 参数
```

## 环境依赖

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- Python 3.11
- PyTorch
- torchvision
- snntorch
- numpy / scipy / pandas / matplotlib

## 数据集

默认 CIFAR-10 数据路径为：

```text
data/cifar10
```

如果数据集已经下载好，训练时不需要额外参数。如果需要让 `torchvision` 自动下载：

```bash
python train.py --data-root data/cifar10 --download
```

## STDP 参数

训练使用当前目录下的：

```text
stdp_params.npz
```

当前参数值为：

```text
A_plus    = 1337.8752754605443
tau_plus  = 4999.999999999999
A_minus   = -249.97810343719524
tau_minus = 4999.999999999999
```

如果需要从器件测试数据重新生成参数文件，可以运行：

```bash
python tools/extract_stdp_params.py --output stdp_params.npz --plot
```

当前项目状态下不建议重新拟合，除非器件数据或拟合方法发生变化。训练、评估和推理命令都应显式传入
`--stdp-params-file stdp_params.npz`。

## 训练

推荐使用稳定的 `conv-stdp` 模型：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/conv_stdp \
  --model-type conv-stdp \
  --augment \
  --scheduler cosine \
  --stdp-pretrain \
  --stdp-epochs 2 \
  --epochs 120 \
  --batch-size 128 \
  --time-steps 10 \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --conv-channels 96 \
  --lr 1e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --device cuda
```

训练结果会保存为：

```text
checkpoints/conv_stdp/last.pt
checkpoints/conv_stdp/best.pt
```

## 评估

评估最优 checkpoint：

```bash
python eval.py \
  --checkpoint checkpoints/conv_stdp/best.pt \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --device cuda \
  --batch-size 256 \
  --seed 42
```

参考结果：

```text
checkpoint=checkpoints/conv_stdp/best.pt
checkpoint_epoch=118
time_steps=10
test_loss=0.9498
test_acc=0.6642
saved_test_loss=0.9519
saved_test_acc=0.6646
```

由于输入使用 Poisson 脉冲编码，单独评估时会有轻微随机波动。可以使用
`--seed 42` 保持结果可复现。

## 推理演示

输出若干测试样本的预测类别和真实类别：

```bash
python infer.py \
  --checkpoint checkpoints/conv_stdp/best.pt \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --device cuda \
  --limit 20
```

## 说明

- `mlp-stdp` 作为对照模型保留，但准确率大约在 `44%` 左右。
- `conv-stdp` 是当前建议汇报和保留的稳定模型。
- 本项目不再继续加入更大的实验模型；继续提升准确率需要额外的模型设计和训练实验。
