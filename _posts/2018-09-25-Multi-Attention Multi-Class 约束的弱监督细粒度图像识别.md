---
layout:     post
title:      用SE结构的Multi-Attention Multi-Class约束
subtitle:   弱监督细粒度图像识别：多个SE结构获得部位的Attention
date:       2018-09-25
author:     BY
header-img: img/tag-bg-o.jpg
catalog: true
tags:
    - 细粒度图像识别
    - Attention
    - SE结构
    - 弱监督信息

---

# 用SE结构的Multi-Attention在同类或不同类上的进行约束_采用NpairLoss弱监督细粒度图像识别
    2018_ECCV,  Ming Sun & Baidu
    用多个SE结构获得部位的Attention，再用N-pair Loss 对这些Attention进行约束。
    使得不同SE结构生成不同的部位Attention，完成弱监督细粒度图像识别。
    还提供了 Dogs-in-the-Wild 数据集。

论文:《[Multi-Attention Multi-Class Constraint for Fine-grained Image Recognition](https://arxiv.org/pdf/1806.05372v1.pdf)》


---

# 引言

现有的基于Attention的细粒度图像识别方法大多都没有考虑 Object part的相关性。而且以往大多方法都用multi-stage 或者 multi-scale机制，导致效率不高切难以end-to-end训练。

此论文提出能调节不同输入图像的不同部位(object-part)的关系:

- 基于SE设计One-squeeze multi-excitation(OSME)方法去学习每张图的每个关注区域特征。
- 再用Multi-attention multi-class constraint(MAMC)方法让同类别图像具有类似的Attention，而不同类别的图像具有不一样的Attention。

此方法效率高且容易end-to-end训练。

此外，论文还提供Dogs-in-the-Wild的综合狗种数据集。它按类别覆盖范围，数据量和注释质量超过了类似的现有数据集。


# 思路


强监督信息的细粒度图像分类的方法依赖于object part 标注，但标注这些信息开销很大，故弱监督信息的方法得到重视。但目前基于弱监督信息的细粒度图像分类的方法或多或少有着下面几个缺点：

- 额外步骤. 如增加额外的regions才能进行part localization 和 feature extraction，这导致计算量增大。
- 训练步骤复杂. 因为构建的结构复杂，需要多次交替或级联训练。
- 单独检测object part，而忽略它们的相关性 (这是最重要的)， 学习到的注意力模块可能只集中在相同的区域。假如好好利用“具有区分性的特征”，可能可以让注意力模块学习出多个不同的部位。

文中提出细粒度图像分类的三个准则：

- 检测的部位应该很好地散布在物体身上，并提取出多种不相关的features
- 提取出的每个part feature都应该能够区分出一些不同的类别
- part extractors（部位检测器）应该能轻便使用。

然后论文就提出了满足这些准则的弱监督信息分类方案：

- Attention用什么？ 用SENet的方案，设计one-squeeze multi-excitation module (OSME)来定位不同的部分。和现有的方法不同，现有的方法大多是裁剪出部位，再用额外的网络结构来前馈该特征。SEnet的输出可以看做soft-mask的作用。
- 如何让Attention学习出多个不同的部位？ 受到度量学习损失的启发（如TribleLoss），提出multi-attention multi-class constraint (MAMC) ，鼓励"same-calss和same-attention"的特征 和 "same-calss和different-attention"的特征 距离更接近。
- 现有的大多方法需要多个前馈过程或者多个交替训练步骤，而论文的方法是端到端的训练。


# 相关工作

在细粒度图像识别的任务中，由于类间差异是微妙的，需要特征的区分学习和对象部位定位。 一种直接的方法是手动在object parts上进行标注，并进行监督学习（强监督信息）。但是获得object part的标注开销巨大，基于弱监督信息的识别机制也就相应被提出。 如STN（空间变换网络）让人脸进行对齐（end2end方式）、如双线性模型( bilinear models)让特征提取和object part定位的两个模型协调合作。 但是论文方法不需要裁剪就可以直接提取object part的特征，切可以高效地拓展到多个object parts。

深度量度学习是学习测量样本对之间的相似性。经典度量学习可认为是学习pairs of points间的 Mahalanobis distance。：

- Siamese网络制定的深度度量学习是：最小化正对之间的距离，同时保持负对分开。
- Triplet Loss 通过优化三个样本中的正对和一个负对的相对距离。虽然在细粒度产品搜索时很有效，但triplet loss 只考虑一个negative sample, 导致训练速度慢，并且易导致局部最小值。 
- N-pair Loss 考虑训练中的多个负样本，使用N元组，缓解了Triple Loss问题。
- Angular Loss 增强了N-pair Loss。考虑角度关系作为相似性度量增加了三阶几何限制，捕获了triplet triangles的附加局部结构，收敛更好。


# 提出的方案

如下图，每个SE结构用于提取不同object-part的soft-mask（即注意的区域），即OSME模块是对每张图片提取P个特征向量（P个object-part的特征）。 

传入多张同类或不同类别的图片，最后得到的Attention传入MAMC中进行学习，让同类的Attention尽量相近，不同类的Attention尽量不一样。

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/MAMC_overview_pig1.png)

如何将提取的Attention特征引导到正确的类标签？ 直接做法是softmax损失对这soft-mask（连续注意力）进行评估。但这方法无法调节Attention features 之间的相关性。 另一类方法是迭代地从粗略生成精细的Attention区域，但由于当前预测依赖于前一次预测，在训练过程中，初始误差会被迭代放大，需要强化学习或者仔细初始化才能较好解决这问题。

而论文提出 multi-attention multi-class constraint (MAMC) 来探索更丰富的object-parts的相关性。

假设现有训练图像$\{(x,y),...\}$ 共 $K$ 张, 和N-pair Loss论文的做法一样，采样出 $N$ 对样本:$B=\{(x_i,x^+_i,y_i),...\}$,其中$x_i$和$x^+_i$都属于$y_i$类，集合 $B$ 中共有 $N$ 类图片。 对于每对样本 $(x_i,x^+_i)$, 用OSME提取出的P个特征向量为: $\{f^p_i,f^{p+}_i\}^P_{p=1}$

对于$y_i$类在第 $p$ 个Attention结构得到的特征向量$f^p_i$ 来说，其他的features可以分为四类：

- same-attention same-class features: $S_{sasc}(f^p_i)=\{f^{p+}_i\}$
- same-attention different-class features: $S_{sadc}(f^p_i)=\{f^p_j, f^{p+}_j\}_{j \neq i}$
- different-attention same-class features: $S_{dasc}(f^p_i)=\{f^q_i, f^{q+}_i\}_{p \neq q}$
- different-attention different-class features: $S_{dadc}(f^p_i)=\{f^q_j, f^{q+}_j\}_{j\neq i , p \neq q}$


那么下面给出那些是需要和$f^p_i$相近的，那些是需要远离$f^p_i$的集合定义，即对于$f^p_i$来说，正负集合的定义：

- Same-attention same-class:
    + $P_{sasc}=S_{sasc},\quad N_{sasc}=S_{sadc} \bigcup S_{dasc}\bigcup S_{dadc}$
- Same-attention different-class:
    + $P_{sadc}=S_{sadc},\quad N_{sadc}=S_{dadc}$
- Different-attention same-class:
    + $P_{dasc}=S_{dasc},\quad N_{dasc}=S_{dadc}$

所以对于任何正例集合$P\in \{P_{sasc},P_{sadc},P_{dasc}\}$ 和负例集合 $N\in \{N_{sasc},N_{sadc},N_{dasc}\}$ , 我们希望当前的部位特征 $f^p_i$ 和正例集合距离越近，而和负例集合距离越远. 令 $m > 0$ 为distance margin，则：
$$
\mid\mid f^p_i - f^+\mid\mid^2 + m \leq \mid\mid f^p_i - f^-\mid\mid,\quad \forall f^+ \in P,\quad f^- \in N
$$

那就可以设计 hinge loss为:
$$\begin{equation*}
\big[\mid\mid f^p_i - f^+\mid\mid^2 -  \mid\mid f^p_i - f^-\mid\mid + m \big]_+
\end{equation*}$$
上述式子虽然广泛采用standard triplet sampling 来优化，但实际中会出现收敛慢、性能不稳定的问题。论文采用2016年出来的N-pair Loss 来优化上式：
$$\begin{equation*}
L^{np} = \frac{1}{N} \sum_{f^p_i \in B}\big\lbrace \sum_{f^+\in P} \log \big(1+\sum_{f^-\in N} exp(f^{pT}_i f^- - f^{pT}_if^+) \big)  \big\rbrace
\end{equation*}$$

那么最终的loss定义为:
$$\begin{equation*}
L^{mamc}=L^{softmax} + \lambda \big(L^{np}_{sasc} + L^{np}_{sadc}+ L^{np}_{dasc} \big)
\end{equation*}$$


# 个人小总结

该方法在 CUB-200-2011、Stanford Dogs、Stanford Cars 以及论文贡献的Dogs-in-the-Wild数据集。 文中介绍了这些数据集，并且列举了许多其他方法在这些数据集上的结果以及细节（如是否采用部件级标注信息、是否能一次性训练完），这些表格数据信息实在是感人。

文中从SENet得到启发，让SE的输出为部位的Attention，期望多个SE结构来分别学习不同部位的Attention。 为了达成这个想法，作者先对Attention出来的特征进行划分正负样例集合，利用了度量学习的N-pair Loss，引导不同的SE结构学习不同的部位。

文中方法是基于弱监督信息的图像识别，文章大体流程清晰明朗。虽未找到论文代码，但复现起来难度应该不会特别大（需要注意训练过程中的N-pair Loss，以及正负例的归属）。


关于N-pair Loss的解释及公式的理解，请移步至下面链接：

- [N-pair Loss论文：2016_nips_Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)
- [理解公式的参考链接：从最优化的角度看待Softmax损失函数](https://zhuanlan.zhihu.com/p/45014864)
- [理解公式的参看链接：LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)


---

若出现格式问题，可移步查看知乎同款文章：[Multi-Attention Multi-Class 约束的弱监督细粒度图像识别](https://zhuanlan.zhihu.com/p/45345038)

---
