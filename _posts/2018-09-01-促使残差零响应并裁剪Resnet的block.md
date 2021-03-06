---
layout:     post
title:      促使残差零响应，并裁剪Resnet的block
subtitle:   设计Relu结构，让残差的输出倾向0
date:       2018-09-01
author:     BY
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - 模型压缩
    - 正则化
    - 稀疏
    - 修剪block

---

# 促使残差零响应，并裁剪Resnet的block
    2018年,  Xin Yu. Utah&NVIDIA
    让ResV2的残差倾向0响应，从而学习严格的IdentityMapping
    剪去冗余零响应残差，让预测网络参数量少且精度不变

论文:《[Learning Strict Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1804.01661v3)》

基于pre-act Resnet，文章中提出的 $\epsilon-ResNet$能自动丢弃产生的响应小于阈值的残差块， 让网络性能略有下降或没有损失。

---


# 引言

Resnet论文中提到 *“我们推测，优化带残差的网络比以前的网络起来更容易。在极端情况下，如果身份映射是最优的，那么将残差被推到零，这比通过一堆非线性层去拟合身份映射更容易”*。

然而实际情况中，深网络并不能完美地让一些残差块的输出接近0，该论文提出促进零残差的方法，即让网络学习严格意义上的身份映射关系。

为了测试深度对残差网络的影响，文中设计从100-300层的残差网络，测试它们在cifar10上的精度。  实验发现随着深度，精度会呈微弱上升的趋势，但波动很大。因此想： 对于训练N层的残差网络，能否找到一个层数远低于N的子网络，让其性能相当？

---

# 本文方法

对于Resnet中，映射函数$H(x)=x+F(x)$.  于是本文作者修改为$H(x)=x+S\big(F(x)\big)$. 其中若$F(x)$ 小于阈值$\boldsymbol{\epsilon}$, 则$S\big(F(x)\big)=\boldsymbol{0}$.如果$F(x)$的响应值都不小，那么$S\big(F(x)\big)=F(x)$ . $S(\cdot)$为sparsity-promoting function.

如果某残差块使用提出的变量产生0响应时，交叉熵项与L2组成的损失函数将把那么它的filter权值推到0。因此在预测时，就可以将这些残差块给删去。

看起来整体思想简单合理，我们拆分得三个子问题：

- 1 如何让$F(x_i)$小于阈值$\boldsymbol{\epsilon_i}$时输出0响应值？
- 2 如何让 对应的残差块参数更新推向0？
- 3 预测时如何删去残差块？（这个其实不算问题了，手动删都可以实现了）


### 问题1：如何让$F(x_i)$小于阈值$\boldsymbol{\epsilon_i}$时输出0响应值

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/4_ReLU_to_zero_or_one.png)
文中采用4个ReLU结构，解决了这个问题（看图）。

- (a,b)表示: $a\cdot F(x_l) + b$ . 参考自论文《Highway Networks》提出的方法。
- 对于$F(x_l)>\epsilon$ 或者 $F(x_l)< - \epsilon$ , $S(F(x_l))= F(x_l)$：
- 对于$\mid F(x_l)\mid < \epsilon$ 则 $S(F(x_l))= 0$.
- $\epsilon$是阈值。$L$是个极大的正数。

> 建议模拟下，这四个ReLU的设计还是挺有意思的. 
> 
> 但笔者觉得还是有点问题，如果$F(x_l)>\epsilon, and \; F(x_l) \rightarrow \epsilon$时，在第三个ReLU中，如何保证$L\cdot\big(F(x_l)- \epsilon\big) + 1 <0 $ ? 这样到最后到第四个ReLU的输出，不能保证输出为1。 
> 
> 当然这也是当$F(x_l)= \epsilon + \sigma$ 且$\sigma < \frac{1}{L}$ 时才会出现问题。
> 不过这已经满足这个结构的需求。
> 
> 其实如果采用Mxnet.gluon框架，用代码: 
> 
> ```
> S = (F > epsilon) * F
> ```
> 
> 就可以实现这个结构功能。 但得到的$S$是ndarray格式，这不是符号流的代码。


### 问题2：如何让 对应的残差块参数更新推向0 ？

当$F(x_l)$特别小时，稀疏促进函数$S$ 的输出为0. 所以这些参数从交叉熵损失函数获得的梯度为0。

但损失函数中还有正则项，正则项会为这些参数提供梯度，从而让这些参数越来越小（推向0）。

残差块中某层的权重直方图如下所示，当残差块被认定可裁剪时，它的参数被迅速收敛到0：

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/weight_collapse.png)


这样所有问题就解决了。


论文还添加了side-supervision，即在损失函数上，除了原先的交叉熵、L2正则项外，还增加了side-loss：

- 假设还存在N个残差块，那么用N/2 个残差块输出来计算side-loss：
    + 对侧面的输出后添加全连接层(最后一层输出维度是类别数)、Softmax、交叉熵。
    + 计算损失后BP回去更新前N/2层的参数
- 在训练期间，缩短前半层的反向传播路径。
- 在总的损失函数中，side-loss的系数设置为0.1。

> 感觉和GoogleNet的做法差不多，但是GoogleNet是为了更好优化网络(梯度消失/爆炸问题)，而此处是想让后N/2个残差块倾向恒等映射（这样就可以删减掉了）。

---

# 总结

$\epsilon-Resnet$ 是标准残差网络的一种变体，可以端到端地自动裁剪无效的冗余残差层。且$\epsilon-Resnet$ 让模型尺寸显著减少，同时在Cifar10/100、SVHN上保持良好精度。


这种简单有效的模型压缩方法还是挺吸引人的，相比其他的采用强化学习、L1正则等等方法，本文方法实现起来简单许多。

文中提到与其特别相似的论文:[Data-driven sparse structure selection for deep neural networks](https://arxiv.org/abs/1707.01213). 类似的还有BlockDrop等.

且还有往其他层次思考：ChannelPruning，如2018_ICLR的《[Rethinking the smaller-norm-less-informative assumption in channel pruning of convolution layers](https://arxiv.org/abs/1802.00124)》

---

若出现格式问题，可移步查看知乎同款文章：[促使残差零响应，并裁剪Resnet的block](https://zhuanlan.zhihu.com/p/42385039)

---

