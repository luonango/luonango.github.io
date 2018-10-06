---
layout:     post
title:      模型压缩：利用BN放缩因子来修剪Channel
subtitle:   用L1将BN的放缩因子推向0
date:       2018-08-01
author:     BY
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - 模型压缩
    - 正则化
    - 稀疏
    - 修剪channels

---


# 模型压缩：利用BN放缩因子来修剪Channel
    2017_ICCV
    利用BN的放缩因子来修剪channel
    用L1将BN的放缩因子推向0.

这次介绍一篇模型压缩的论文： **将预训练好的网络删减掉一些Channel（再fine-tuning），让模型参数减少的同时，还能让准确率维持不变（或精度损失很少）。**

论文为:《[Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/abs/1708.06519)》

---

那问题来了：

- 1）那它是以什么准则来删减Channel？
- 2）总体训练步骤是什么？
- 3）效果如何？优缺点是？
- 4）类似相关工作有哪些？

---


论文方法从BN中得到启发。我们回顾下BN：
$$
\hat{z} = \frac{z_{in} - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} ;\quad z_{out} = \gamma \hat{z} + \beta
$$

>其中$\mu_B$表示mini-batch B中某feature map的均值。
>
> scale $\gamma$ 和 shift $\beta$ 都是通过反向传播训练更新。


这么看来，可以直接用 $\gamma$ 来评估channel的重要程度。$\gamma$ 的数越小，说明该channel的信息越不重要，也就可以删减掉该Channel。

> 此处笔者有点疑问，为什么忽略 $\beta$ 作为评估因子？
> 
> 笔者猜想的答案为： 
> 
> - feature map的重要性是看其方差而非均值。方差越大则该feature map内的特征就越明显。而均值对feature map内部的特征表达无太大关系。
> - 由于 $z_{out} \sim N(\beta,\gamma^2)$，$\gamma$ 即方差越小，则该feature map也就可判定为重要性越小。
> - 下一层($l+1$层)的第 $j$ 张$map_{l+1,j}$的值，是其卷积核们对 $l$ 层的所有 $map_{l,:}$ 进行卷积求和。所以如果某 $map_{l,i}$ 的方差特别小（意味着 $map_{l,i}$ 里面的所有值都相近），那么这个 $map_{l,i}$ 对 $map_{l+1,j}$ 上所有单元的值的贡献都是一样的。
> - 这么看来，就算去掉了 $map_{l,i}$， $map_{l+1,j}$ 内的特征变化还是不大（即只是分布平移，而不是发生拉伸等变化，神经元之间的差异性变化不大）。
> 


虽然可以通过删减 $\gamma$值接近零的channel，但是一般情况下，$\gamma$值靠近0的channel还是属于少数。于是作者采用smooth-L1 惩罚 $\gamma$ ，来让$\gamma$值倾向于0。


那么网络的损失函数就可设计为:
$$
L=\sum_{(x,y)} l\big(f(x,W),y\big) + \lambda \sum_{\gamma \in \Gamma} g(\gamma)
$$
其中$(x,y)$是训练的输入和目标， $W$是可训练的权重，$g(\cdot)$ 是引导稀疏的惩罚函数，$\lambda$作为这两项的调整。 文中选择$g(\cdot)=\mid s\mid$,当然也可以采用Smooth_L1方法在零点为光滑曲线.

> Tips：
> 
> - 论文中提到Smooth-L1时,引用的是：[2007_Fast optimization methods for l1 regularization: A comparative study and two new approaches](https://link.springer.com/chapter/10.1007/978-3-540-74958-5_28)
>
> - 而[2015_Fast R-CNN](https://arxiv.org/abs/1504.08083) 提出了 Smooth-L1 公式为:
> $$ 
smooth_{L1} (x)=\left\{
             \begin{array}{l}
             0.5 x^2, & if\; \mid x\mid < 1 \\
             \mid x\mid - 0.5 , & o.w. 
             \end{array}\right.
$$ 
> - 作者源码采用的不是Fast R-CNN提出的SmoothL1. 可以看下[论文提供的源码](https://github.com/liuzhuang13/slimming)

---


接下来我们看看训练过程（非常简明的步骤）：

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/NetworkSlimming_1.png)

- 第一步：初始化网络；
- 第二步：加入Channel稀疏惩罚项，训练网络；
- 第三步：通过固定阈值来删减channel，如删减70%的channel；
- 第四步：Fine-tune。由于删减channel后精度会下降，故再训练去微调网络；
- 第五步：可以再跳到第二步，实现多次精简网络；
- 第六步：得到精简后的网络。

方法简单明了，实现起来也是很方便容易。

论文采用VGG、DenseNet、Resnet模型，在数据集CIFAR10、CIFAR100、SVHN以及ImageNet上进行实验。

结果表明此方法在参数量在保持相似精度情况下，参数瘦身最高可达到10倍，且计算量上也能大大减少。

更多的实验及优缺点分析请详细阅读[原论文](https://arxiv.org/abs/1708.06519)。

---

说完了文章的思路及方法，我们再温习（预习）下论文的相关工作吧。

模型压缩的工作有（想知道下述方法的论文名，请查阅本论文的参考文献）：

- Low-rank Decomposition 低秩分解：
    + 使用SVD等技术来近似于权重矩阵（它具有低秩矩阵）。
    + 在全连接层上工作很好，但CNN的计算主要在卷积层。
- Weight Quantization 量化权值：
    + 如HashNet量化网络权值（采用共享权重和哈希索引大大节省存储空间）
    + 但不能节省运行时间（因为权重还需要恢复从而进行网络推理inference）
    + 二值化是个很好的方法（或用三值化{-1,0,1}），但它会有精度损失。
- Weight Pruning/Sparsifying 权重修剪或稀疏：
    + 有论文将训练好的网络里的小权值修剪掉（即设为0），这样也可以用稀疏格式储存权值。
    + 但是需要专用的稀疏矩阵运算库或特殊硬件来加速，且运行内存也没有减少。
- Structured Pruning/Sparsifying 结构修剪或稀疏化：
    + 有提出在训练好的网络中，修剪那些较小权值连接的Channel，再微调网络恢复精度方法的论文
    + 有提出在训练前随机停用channel从而引入稀疏，但这会带来精度损失。
    + 有提出neuron-level的稀疏方法从而修剪神经元获得紧凑玩了个，也有提出结构化稀疏学习（SSL）的方法，去稀疏CNN不同层级的结构（filters、channels、layers）。但是这些方法在训练期间均采用群组稀疏正则(Group sparsity Regualarization)方法来获得结构正则，而本文采用简单的L1稀疏，优化目标要简单很多。
- Neural Architecture Learning 神经结构学习：
    + 有关于自动学习网络结构的方法，如谷歌的几篇通过强化学习来搜寻最佳网络结构，或者其他的给定巨大网络结构，从中学习出最佳子图网络。
    + 但是资源消耗太大，时间太长。

---

若出现格式问题，可移步查看知乎同款文章：[模型压缩：利用BN放缩因子来修剪Channel](https://zhuanlan.zhihu.com/p/39761855)