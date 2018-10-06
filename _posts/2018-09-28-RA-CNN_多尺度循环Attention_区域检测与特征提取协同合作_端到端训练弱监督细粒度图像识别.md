---
layout:     post
title:      RA-CNN 多尺度循环Attention
subtitle:   细粒度图像识别：区域检测与特征提取协同合作
date:       2018-09-28
author:     BY
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 细粒度图像识别
    - Attention
    - 区域检测
    - Attention mask

---

# RA-CNN_多尺度循环Attention_区域检测与特征提取协同合作_端到端训练弱监督细粒度图像识别
    2017_CVPR,  Jianlong Fu & Heliang Zheng & Tao Mei
    微软亚洲研究院
    多尺度循环Attention网络，让区域检测与特征提取相互协作提升。
    提出APN注意力建议子网络（attention proposal sub-network)
    循环精调出Attention区域位置（可以并入反向传播，端到端训练）

论文:《[Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition](http://202.38.196.91/cache/9/03/openaccess.thecvf.com/c9a21b1be647ad791694df3a13879276/Fu_Look_Closer_to_CVPR_2017_paper.pdf)》

---

# 前言

细粒度物体识别的难点在于 **判别区域定位(discriminative region localization) 和 基于区域的特征表达(fine-grained feature learning)** 。

一般part-based的做法分为两步走：

- 1 通过无监督或强监督(指利用box/part标注信息)的方法，分析卷积层的响应，再定位出object区域。
- 2 提取所有判别区域的特征进行encode，再拿去做识别。

但是这样有些弊端：

- 1 人类标注的box/part 信息 或无监督得到的part区域，未必就是最最最适合机器用于识别的信息（非最优解）。
- 2 对于相似类别的一些区域，细微的视觉差异仍然难以学习并区分。

并且作者发现区域检测和细粒度特征学习是相互关联的，因此它们可以相互加强。

为了解决上述难点及弊端，作者提出无需box/part 标注信息的RA-CNN（recurrent attention CNN)，递归地学习判别区域Attention和基于区域的特征表示，并让它们协同强化。


#　论文做法：

### 网络设计：

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/RA-CNN_architecture_pig1.png)

- 多尺度网络：
    + 共享相同的网络结构，但每个尺度的网络有各自的参数，从而适应不同分辨率（不同尺度）的图。
    + 如图有三个尺度的网络（scale1-3), b1-3表示卷积网络。
    + 文中采用VGG19或VGG16作为网络结构。均记载在Imagenet上预训练好的参数。
- 分类子网络：
    + 全连接层 + softmax层，c1-c3即为三个分类子网络。
    + 和平时的最后分类结构一样。
- APN（Attention proposal sub-network):
    + 是FC结构，传入网络的输出，输出三个数值 $(t_x,t_y,t_l)$ . 其中$t_x,t_y$表示 区域框的中心点，$t_l$ 表示框边长的一半（为正方形框）。
    + 与其他的方法不同，此处是通过学习得到框的位置及大小。
- Crop and Zoom in 操作：
    + 结合APN的输出与前一尺度的图，得到区域框的图后再进行放大。
    + 目的为了提取更细粒度的特征
    + 放大到与scal1的尺度相同，这样网络的输入尺寸也就对应上了。
- Loss 问题留在后面再讲。

乍这么一看，似乎没什么问题，甚至会有“我们也能想到这方法阿”这样的冲动。实际上隐藏了个大boss：“得到APN的输出后，如何在裁剪放大图片的同时，还能保证反向传播顺利完成？”

(思考 ing....　


---

(好的，我放弃了，看作者怎么变魔术吧...

论文用Attention mask来近似裁剪操作。 一提到mask 就应该想到mask乘上原图能得到区域图。对的，论文就是定义一个连续的mask函数$M(\cdot)$ ，让反向传播顺利完成。 这个函数$M(\cdot)$ 是二维boxcar函数的变体。


那么裁剪后的图片结果就是($X$为前一尺度的图,$\odot$表示逐元素乘)：
$$
x^{att}=X \odot M(t_x,t_y,t_l)
$$

令图的左上角(top-right,tl)点为$t_l$ ,右下角(bottom-right,br)。那么APN输出框的左上角$(t_{x(tl)},t_{y(tl)})$和右下角$(t_{x(br)},t_{y(br)})$ 为：
$$
t_{x(tl)}=t_x - t_l,\quad t_{y(tl)}=t_y - t_l \\
t_{x(br)}=t_x + t_l,\quad t_{y(br)}=t_y + t_l
$$

则连续的mask函数$M(\cdot)$ 为：
$$
M(\cdot) = \big[h\big(x-t_{x(tl)}\big) - h\big(x-t_{x(br)}\big) \big]\cdot \big[h\big(y-t_{y(tl)}\big) - h\big(y-t_{y(br)}\big) \big]
$$
其中里面的$h(\dot)$是logistic 函数($k=1$时即为sigmoid函数):
$$
h(x)=\frac{1}{1+ \exp^{-kx}}
$$

当$k$足够大时，这个逻辑函数就可认为是step function(阶梯函数)。换句话说，如果点$(x,y)$ 在框内，则 M 对应的值近似为1，否则近似为0。 这样的性质跟二维boxcar函数一样，可以很好地近似裁剪操作。文中让$k=10$.

接下来是自适应缩放到大尺寸的做法了。论文采用双线性插值来计算目标图$X^{amp}$的点（利用$X^{att}$里对应的四个邻近点）。

---

### Loss函数

作者为了交替产生准确的区域Attention，切学习出更精细的特征，定义了两种损失函数：$L_{cls}$（intra-scale classification loss，尺度内分类损失）和$L_{rank}$(inter-scale pairwise ranking loss​,尺度间排序损失):

$$
L(X)=\sum^3_{s=1}\big\{L_{cls}\big(Y^{(s)},Y^*\big)\big\} + \sum^2_{s=1}\big\{L_{rank}\big(p^{(s)}_t, p^{(s+1)}_t\big)\big\}
$$

其中$p^{(s)}_t$表示s尺度网络的softmax预测向量中对应正确标签$t$的概率值（上面的总架构图有标出）。且$L_{rank}$定义为： 
$$
L_{rank}\big(p^{(s)}_t, p^{(s+1)}_t\big) = \max \big\{0,p^{(s)}_t-p^{(s+1)}_t + margin\big\}
$$
>强迫$p^{(s+1)}_t > p^{(s)}_t + margin$, 让细粒度尺寸的网络以粗尺度网络的预测作为参考，强制细尺度的网络逐步定位出最具判别力的区域，并产生更高的预测值。论文中让margin=0.05。


最后还有个多尺度特征融合，即让多个尺度网络的最终输出（softmax前一层）进行concat，再并入FC+softmax 得到最终预测值。

---

### 训练步骤(交替训练)：

- 1 用在Imagenet上预训练好的VGG网络参数来初始化分类子网络中的卷积层与全连接层的参数
- 2 初始化APN参数，用分类子网络最后一层卷积层中具有高响应值的区域来初始化$(t_x,t_y,t_l)$
- 3 固定APN的参数，训练分类子网络直到$L_{cls}$收敛
- 4 固定分类子网络的参数，训练APN网络直到$L{rak}$收敛
- 5 交替循环步骤3、4，直到两个网络损失都收敛。

---

此外，论文还仔细分析了关于Attention Learning问题，即为什么Attention mask函数能正确更新 $(t_x,t_y,t_l)$的问题。并且论文中还通过导数图(derivative)分析了不同情况下 $(t_x,t_y,t_l)$它们各自的更新方向。
>导数图:  计算导数范数的负平方(negative square of the norm of the derivatives), 获得与人类感知一致的优化方向. 参考自[2015_ECCV_The Unreasonable Effectiveness of Noisy Data for Fine-Grained Recognition](https://arxiv.org/abs/1511.06789)



# 小小总结

论文提出的RA-CNN，无需box/part 标注信息就能挺好地学习判别区域Attention和基于区域的特征表示，最后得出来的Attention 区域如APN1输出的裁剪框和人类标注的物体框特别接近，同时最终分类结果甚至优于其他基于强监督信息的方法。

让我惊艳的是论文采用Attention mask来近似裁剪操作，从而让反向传播顺利进行。并且$L_{rank}$(inter-scale pairwise ranking loss​,尺度间排序损失)的设计也很巧妙合理。这些解决问题思路方法都值得好好思考思考。

同时论文提到的导数图（导数范数的负平方与人类感知的优化方向一致），这个我还是第一次听说，有时间得好好理解学习下。