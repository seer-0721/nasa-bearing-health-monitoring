# CIFAR-10 STDP SNN

本项目使用器件实验拟合得到的 STDP 参数，在 CIFAR-10 数据集上训练脉冲神经网络。
训练时必须读取当前目录下的 `stdp_params.npz`。这是本项目的硬性要求；
如果文件不存在，程序会直接报错，不会自动改用其他参数文件。

当前稳定 STDP 版本使用 `conv-stdp` 模型，在 CIFAR-10 测试集上约为 `66%` 准确率：

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

## 推荐的 SNN + STDP 训练

如果任务要求必须使用 SNN + STDP，优先使用 `spiking-conv-stdp`。这个模型保留 CIFAR-10 的脉冲
输入编码，卷积特征提取也使用 LIF 放电，然后接入器件参数控制的 STDP synapse 和 LIF 分类头。
它比旧的 `conv-stdp` 更符合脉冲网络约束，也比 MLP/浅卷积基线有更大的容量。

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/spiking_conv_stdp_deterministic \
  --model-type spiking-conv-stdp \
  --augment \
  --scheduler cosine \
  --stdp-pretrain \
  --stdp-epochs 2 \
  --epochs 160 \
  --batch-size 128 \
  --time-steps 16 \
  --encoding deterministic-rate \
  --eval-repeats 1 \
  --hidden-size1 768 \
  --hidden-size2 384 \
  --conv-channels 128 \
  --dropout 0.1 \
  --beta 0.9 \
  --threshold 1.0 \
  --lr 1e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --amp \
  --device cuda
```

评估：

```bash
python eval.py \
  --checkpoint checkpoints/spiking_conv_stdp_deterministic/best.pt \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --device cuda \
  --batch-size 256
```

`deterministic-rate` 编码是确定性的，评估时不需要重复采样。只有使用 `poisson` 编码时才建议设置
`--eval-repeats` 对同一个样本做多次脉冲编码并平均 logits。
如果显存不足，可以把 `--batch-size` 降到 `64`；如果训练太慢，可以先把 `--time-steps` 降到 `10` 做快速对比。

当前最好结果：

```text
checkpoint=checkpoints/spiking_conv_stdp_deterministic/best.pt
checkpoint_epoch=79
encoding=deterministic-rate
saved_test_acc=0.7735
reeval_test_acc=0.7739
```

已淘汰实验：

```text
AutoAugment + RandomErasing(0.25) + CutMix(1.0)
epoch=023 test_acc=0.1813

dropout=0.2 + weight_decay=1e-3
epoch=089 test_acc=0.7791

cifar-resnet teacher distillation, alpha=0.5, T=4.0
epoch=072 test_acc=0.7492
```

强增强对当前 SNN/STDP 训练过强，不建议继续。单纯提高 dropout/weight decay 没有带来有效提升。
使用 94% ResNet teacher 做蒸馏也低于 baseline，当前不作为主线。

## 下一轮提升实验

当前实验显示：`spiking-conv-stdp` 训练准确率可到约 `90%`，但测试准确率大约在 `80%` 附近波动。
小正则、强增强和蒸馏都没有明显解决问题。下一轮优先尝试硬件 STDP 校准类方法，而不是继续做普通
regularization。

### 方法 1：软件补偿 LTD

器件参数中 LTP 明显强于 LTD。`--stdp-ltd-scale` 会在软件层面对 LTD 项乘一个系数：

```text
delta_w = A_plus * LTP - A_minus * stdp_ltd_scale * LTD
```

先试 `4.0`，不要一开始就和 WTA 混用：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/spiking_conv_stdp_deep_ltd4 \
  --model-type spiking-conv-stdp \
  --augment \
  --scheduler cosine \
  --stdp-pretrain \
  --stdp-epochs 2 \
  --epochs 160 \
  --batch-size 128 \
  --time-steps 16 \
  --encoding deterministic-rate \
  --eval-repeats 1 \
  --hidden-size1 768 \
  --hidden-size2 384 \
  --conv-channels 128 \
  --conv-blocks-per-stage 2 \
  --dropout 0.1 \
  --readout spike \
  --beta 0.9 \
  --threshold 1.0 \
  --stdp-ltd-scale 4.0 \
  --lr 1e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --amp \
  --device cuda
```

如果 `4.0` 比 baseline 好，再试 `--stdp-ltd-scale 6.0`。如果前 `30-50` epoch 明显低于 baseline，
直接停止。

### 方法 2：WTA / lateral inhibition

`--wta-k` 会在 STDP hidden layer 上保留每个样本电流最高的 `k` 个 spiking neurons，近似 lateral
inhibition。`--adaptive-threshold-scale` 会对最近发放过的 hidden neurons 增加临时阈值，避免少数神经元
长期垄断。

先试一个温和配置：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/spiking_conv_stdp_deep_wta64 \
  --model-type spiking-conv-stdp \
  --augment \
  --scheduler cosine \
  --stdp-pretrain \
  --stdp-epochs 2 \
  --epochs 160 \
  --batch-size 128 \
  --time-steps 16 \
  --encoding deterministic-rate \
  --eval-repeats 1 \
  --hidden-size1 768 \
  --hidden-size2 384 \
  --conv-channels 128 \
  --conv-blocks-per-stage 2 \
  --dropout 0.1 \
  --readout spike \
  --wta-k 64 \
  --adaptive-threshold-scale 0.2 \
  --adaptive-threshold-decay 0.9 \
  --beta 0.9 \
  --threshold 1.0 \
  --lr 1e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --amp \
  --device cuda
```

如果 WTA 版起步太慢或 test accuracy 掉到随机附近，把 `--wta-k` 放宽到 `128`，或者先去掉
`--adaptive-threshold-scale`。

## 编码方式小批次对比

编码方式对 SNN + STDP 影响很大。当前 checkpoint 证据显示，`deterministic-rate` 是主线：
`checkpoints/spiking_conv_stdp_deterministic/best.pt` 在 epoch 79 达到约 `77.35%`，而同目录结构下
`poisson` 的 `checkpoints/spiking_conv_stdp/best.pt` 只保存到约 `66.66%`。因此完整训练优先使用
`deterministic-rate`，`poisson` 和 `latency` 只作为 ablation 参考。

如果继续做快速实验，建议先固定学习率，不要在 15 个 epoch 内把 cosine 学习率衰减到 `1e-5`：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/encoding_deterministic_fixed_lr \
  --model-type spiking-conv-stdp \
  --augment \
  --stdp-pretrain \
  --stdp-epochs 1 \
  --epochs 30 \
  --batch-size 128 \
  --time-steps 12 \
  --encoding deterministic-rate \
  --eval-repeats 1 \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --conv-channels 96 \
  --dropout 0.1 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --limit-train 10000 \
  --limit-test 2000 \
  --amp \
  --device cuda
```

也建议在旧的稳定 `conv-stdp` 基线上试同一个 deterministic 编码，判断“编码改进”是否能直接抬高原来的
`66%` 基线：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/conv_stdp_deterministic_small \
  --model-type conv-stdp \
  --augment \
  --stdp-pretrain \
  --stdp-epochs 1 \
  --epochs 30 \
  --batch-size 128 \
  --time-steps 12 \
  --encoding deterministic-rate \
  --eval-repeats 1 \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --conv-channels 96 \
  --lr 1e-3 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --limit-train 10000 \
  --limit-test 2000 \
  --amp \
  --device cuda
```

下面三条命令保留作为编码 ablation，对比时除 `--encoding` 和 `--checkpoint-dir` 外保持一致。

Poisson baseline：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/encoding_poisson_small \
  --model-type spiking-conv-stdp \
  --augment \
  --scheduler cosine \
  --stdp-pretrain \
  --stdp-epochs 1 \
  --epochs 15 \
  --batch-size 128 \
  --time-steps 12 \
  --encoding poisson \
  --eval-repeats 1 \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --conv-channels 96 \
  --dropout 0.1 \
  --lr 1e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --limit-train 10000 \
  --limit-test 2000 \
  --amp \
  --device cuda
```

Deterministic rate：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/encoding_deterministic_small \
  --model-type spiking-conv-stdp \
  --augment \
  --scheduler cosine \
  --stdp-pretrain \
  --stdp-epochs 1 \
  --epochs 15 \
  --batch-size 128 \
  --time-steps 12 \
  --encoding deterministic-rate \
  --eval-repeats 1 \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --conv-channels 96 \
  --dropout 0.1 \
  --lr 1e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --limit-train 10000 \
  --limit-test 2000 \
  --amp \
  --device cuda
```

Latency / TTFS：

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/encoding_latency_small \
  --model-type spiking-conv-stdp \
  --augment \
  --scheduler cosine \
  --stdp-pretrain \
  --stdp-epochs 1 \
  --epochs 15 \
  --batch-size 128 \
  --time-steps 12 \
  --encoding latency \
  --eval-repeats 1 \
  --hidden-size1 512 \
  --hidden-size2 256 \
  --conv-channels 96 \
  --dropout 0.1 \
  --lr 1e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --label-smoothing 0.05 \
  --limit-train 10000 \
  --limit-test 2000 \
  --amp \
  --device cuda
```

如果某个编码明显更快到达更高 `test_acc`，再去掉 `--limit-train/--limit-test` 做完整训练。

## 非 SNN 参考上限

`cifar-resnet` 是普通 ANN/ResNet 参考模型，不使用 Poisson 编码，也不是 STDP/SNN 模型。它可以用来确认
数据集、增强和训练环境没有问题，但不应作为 SNN + STDP 分类任务的最终结果。

```bash
python train.py \
  --data-root data/cifar10 \
  --stdp-params-file stdp_params.npz \
  --checkpoint-dir checkpoints/cifar_resnet \
  --model-type cifar-resnet \
  --epochs 200 \
  --batch-size 128 \
  --augment \
  --auto-augment \
  --random-erasing 0.25 \
  --scheduler cosine \
  --lr 3e-3 \
  --min-lr 1e-5 \
  --weight-decay 5e-4 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.2 \
  --resnet-width 64 \
  --resnet-blocks 3 \
  --dropout 0.1 \
  --amp \
  --device cuda
```

## STDP 基线训练

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
- `conv-stdp` 是旧的稳定 STDP/SNN 基线，准确率约为 `66%`。
- `spiking-conv-stdp` 是当前建议继续调优和汇报的 SNN + STDP 主模型。
- `cifar-resnet` 只适合做非 SNN 参考上限；它不是 STDP/SNN 模型。
