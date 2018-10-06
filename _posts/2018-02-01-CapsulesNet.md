---
layout:     post
title:      CapsulesNet的解析
subtitle:   
date:       2018-02-01
author:     BY
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - iOS
    - ReactiveCocoa
    - 函数式编程
    - 开源框架
---

# CapsulesNet的解析

## 前言：

本文先简单介绍传统CNN的局限性及Hinton提出的Capsule性质，再详细解析Hinton团队近期发布的基于动态路由及EM路由的CapsuleNet论文。

## Hinton对CNN的思考

Hinton认为卷积神经网络是不太正确的，它既不对应生物神经系统，也不对应认知神经科学，甚至连CNN本身的目标都是有误的。

#### 在生物神经系统上不成立

- 反向传播难以成立。神经系统需要能够精准地求导数，对矩阵转置，利用链式法则，这种解剖学上从来也没有发现这样的系统存在的证据。
- 神经系统含有分层，但是层数不高，而CNN一般极深。生物系统传导在ms量级（GPU在us量级），比GPU慢但效果显著。
- 灵长类大脑皮层中大量存在皮层微柱，其内部含有上百个神经元，并且还存在内部分层。与我们使用的CNN不同，它的一层还含有复杂的内部结构。

#### 在认知神经科学上CNN也靠不住脚
人会不自觉地根据物体形状建立“坐标框架”(coordinate frame)。并且坐标框架的不同会极大地改变人的认知。人的识别过程受到了空间概念的支配，判断物体是否一样时，我们需要通过旋转把坐标框架变得一致，才能从直觉上知道它们是否一致，但是CNN没有类似的“坐标框架”。如人类判断下图两字母是否一致：
![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/Capsules_net_Mental_rotation.png)
 
#### CNN的目标不正确

先解释下不变性（Invariance)和同变性（Equivariance）。不变性是指物体本身不发生任何变化；同变性指物体可以发生变化，但变化后仍是这种物体。如将MNIST图片中的数字进行平移（该数字形状无变化），就实现了该数字的不变性。如果数字进行了立体的旋转翻转等，但人类仍能识别出这个数字，这就是该数字的同变性。

显然，我们希望的是CNN能够实现对物体的同变性（Equivariance），虽然CNN在卷积操作时实现了同变性（卷积操作是对视野内的物体进行矩阵转换，实现了视角变换），但主要由于Pooling，从而让CNN引入了不变性。这从而让CNN实现对物体的不变性而不是同变性。

比如CNN对旋转没有不变性（即旋转后的图片和原图CNN认为是不一样的），我们平时是采用数据增强方式让达到类似意义上的旋转不变性（CNN记住了某图片的各种角度，但要是有个新的旋转角度，CNN仍会出问题）。当CNN对旋转没有不变性时，也就意味着舍弃了“坐标框架”。

虽然以往CNN的识别准确率高且稳定，但我们最终目标不是为了准确率，而是为了得到对内容的良好表示，从而达到“理解”内容。

## Hinton提出的Capsules 

基于上述种种思考，Hinton认为物体和观察者之间的关系（比如物体的姿态），应该由一整套激活的神经元表示，而不是由单个神经元或者一组粗编码（coarse-coded）表示（即一层中有复杂的内部结构）。这样的表示，才能有效表达关于“坐标框架”的先验知识。且构成的网络必须得实现物体同变性。

这一套神经元指的就是Capsule。Capsule是一个高维向量，用一组神经元而不是一个来代表一个实体。并且它还隐含着所代表的实体出现的概率。

Hinton认为存在的两种同变性(Equivariance)及capsule的解决方法：

- 位置编码（place-coded）：视觉中的内容的位置发生了较大变化，则会由不同的 Capsule 表示其内容。
- 速率编码（rate-coded）：视觉中的内容为位置发生了较小的变化，则会由相同的 Capsule 表示其内容，但是内容有所改变。
- 两者的联系是，高层的 capsule 有更广的域 (domain)，所以低层的 place-coded 信息到高层会变成 rate-coded。


---

## 第一篇《Dynamic Routing Between Capsules》的解析

好的，那让我们看看Hinton 团队2017年10月公布的论文：[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)（Sara Sabour为第一作者）

官方源代码已发布：[Tensorflow代码](https://github.com/Sarasra/models/tree/master/research/capsules) 。不得不提下，由于官方代码晚于论文，论文发布后很多研究者尝试复现其代码，获得大家好评的有[HuadongLiao的CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)、[Xifeng Guo的CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)等等。

该论文中的Capsule是一组神经元（即向量），表示特定类型实体的实例化参数。而其向量的长度表示该实体存在的概率、向量的方向表示实例化的参数。同一层的 capsule 将通过变换矩阵对高层的 capsule 的实例化参数进行预测。当多个预测一致时（文中使用动态路由使预测一致），高层的 capsule 将变得活跃。

### 动态路由方法

有很多方法实现capsule的总体思路，该论文只展示了动态路由方法（之后不久Hinton用EM-Routing的方法实现capsule的整体思路，相应解析在后面）。 

由于想让Capsule的输出向量的长度，来表示该capsule代表的实体在当前的输入中出现的概率，故需要将输出向量的长度（模长）限制在$[0,1]$。文中采用Squashing的非线性函数作为激活函数来限制模长，令当前层是$j$层，$v_j$为$capsule_j$的输出向量，$s_j$是$capsule_j$的所有输入向量。
$$
v_j=\frac{\mid\mid s_j\mid \mid ^2}{1+\mid\mid s_j\mid\mid ^2}\frac{s_j}{\mid\mid s_j\mid \mid }
$$
那$s_j$怎么来呢？别急，我们定义一些概念先：

令$u_i$是前一层(第$i$层）所有$capsule_i$的输出向量，且$\hat{u}_{j|i}$是$u_i$经过权重矩阵$W_{ij}$变换后的预测向量。
那除了第一层，其他层的$s_j$都是前一层所有$capsule_i$的预测向量$\hat{u}_{j|i}$的加权和（加权系数为$c_{ij}$）。
$$
s_j = \sum_i c_{ij}\hat{u}_{j|i} \quad,\qquad \hat{u}_{j|i} = W_{ij}u_i
$$
其中的耦合系数$c_{ij}$就是在迭代动态路由过程中确定的，且每个$capsule_i$与高层所有$capsule_j$的耦合系数$c_{ij}$之和为1。它是通过‘路由softmax’得出：
$$
c_{ij}=\frac{exp(b_{ij})}{\sum_k exp(b_{ik})}
$$
$b_{ij}$可理解为当前$j$层的输出向量$v_j$与前一层所有$capsule_i$的预测向量$\hat{u}_{j|i}$的相似程度，它在动态路由中的迭代公式为：
$$
b_{ij}=b_{ij}+\hat{u}_{j|i}v_j
$$

那么，当得到$l$层的$capsule_i$所有输出向量$u_i$，求$l+1$层的输出向量的流程为：

- a) 通过权重矩阵$W_{ij}$，将$u_i$变换得到预测向量$\hat{u}_{j|i}$。权重矩阵$W$是通过反向传播更新的。
- b) 进入动态路由进行迭代（通常迭代3次就可以得到较好结果）:
    + ![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/Capsules_net_6.png)
- c) 得到第$l+1$ 层$capsule_j$的输出向量$v_j$

可以看看[机器之心绘制的层级结构图](https://www.jiqizhixin.com/articles/2017-11-05)来加深理解：
![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/CapsNet_1.jpg)

### Capsule网络结构
解决了动态路由的算法流程后，我们再来看下论文设计的简单的网络结构：
![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/Capsules_net_1.png)

- **第一层ReLU Conv1层的获取**:
    + 普通的卷积层方法，使用256个9×9的卷积核、步幅为1、ReLU作为激活函数，得到256×20×20的张量输出。与前面的输入图片层之间的参数数量为:256×1×9×9+256=20992. 
- **第二层PrimaryCaps层的获取**:
    + 从普通卷积层构建成Capsule结构的PrimaryCaps层，用了256×32×8个9×9的卷积核，步幅为2，32×8个bias，得到32×8×6×6的张量输出（即32×6×6个元素数为8的capsule）。
    + 与前面层之间的参数总量为：256×32×8×9×9+32×8=5308672.（不考虑routing里面的参数）
    + 值得注意的是，官方源码中接着对这32×6×6个capsule进行动态路由过程（里面包括了Squashing操作），得到新的输出。而论文中没提及到此处需要加上动态路由过程，导致一些研究人员复现的代码是直接将输出经过Squashing函数进行更新输出（部分复现者已在复现代码中添加了动态路由过程）。
    + PrimaryCaps层的具体构建可查看源码中的layers.conv_slim_capsule函数。
- **第三层DigitCaps层的获取**:
    + 从PrimaryCaps层有32*6*6=1152个capsule，DigitCaps有10个，所以权重矩阵$W_{ij}$一共有1152×10个，且$W_{ij}=[8×16]$，即$i \in [0,8), j\in[0,16)$。另外还有10*16个偏置值。
    + 进行动态路由更新，最终得到10*16的张量输出。
- **参数的更新**：
    + 权重矩阵 $W_{ij}、bias$通过反向传播进行更新。
    + 动态路由中引入的参数如$c_{ij}、b_{ij}$均在动态路由迭代过程中进行更新。

### 损失函数
解决了论文设计的网络结构后，我们来看下论文采用损失函数（Max-Margin Loss形式）：
$$
L_c=T_c \max\big(0,m^+ - \mid\mid v_c\mid\mid\big)^2 + \lambda\big(1-T_c\big) \max\big(0,\mid\mid v_c\mid\mid - m^-\big)^2 
$$

- 其中$c$表示类别，$c$存在时，指示函数$T_c=1$,否则$T_c=0$。$m^+、m^-$分别为上边界和下边界，$\mid\mid v_c\mid\mid$为$v_c$的L2范数。论文设置$\lambda=0.5$，降低第二项的loss的系数，防止活泼的（模长较大）capsule倾向于缩小模长，从而防止网络训练差（系数大则求导的数值绝对值大，则第二项loss反馈的更新会更有力）。上下边界一般不为1或0，是为了防止分类器过度自信。
- 总loss值是每个类别的$L_c$的和:  $L=\sum_{c} L_c$
- 该损失函数与softmax区别在于：
    + softmax倾向于提高单一类别的预测概率而极力压低其他类别的预测概率，且各类别的预测概率和为1。适用于单类别场景中的预测分类。
    + 而此损失函数，要么提高某类别的预测概率（若出现了该类 ），要么压低某类别的预测概率（若未出现该类），不同类别间的预测概率互不干扰，每个类别的预测概率均在$[0,1]$中取值。适用于多类别并存场景中的预测分类。

### 重构与表征

重构的思路很简单，利用DigitCaps中的capsule向量，重新构建出对应类别的图像。文章中使用额外的重构损失来促进 DigitCaps 层对输入数字图片进行编码：
![](![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/Capsules_net_2.png)

由于DigitCaps中每个capsule向量都代表着一个实体(数字)，文章采用掩盖全部非正确数字的capsule，留下代表正确数字实体的capsule来重构图片。

此处的重构损失函数是采用计算在最后的 FC Sigmoid 层采用的输出像素点与原始图像像素点间的欧几里德距离。且在训练中，为了防止重构损失主导了整体损失（从而体现不出Max-Margin Loss作用），文章采用 0.0005 的比例缩小了重构损失。

### 实验结果

#### 采用MNIST数据集进行分类预测：
在此之前，一些研究者使用或不使用集成+数据增强，测试集错误率分别为0.21%和0.57%。而本文在单模型、无数据增强情况下最高达到0.25%的测试集错误率。具体如下图：
![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/Dynamic_Routing_10.png)

#### 采用MultiMNIST数据集进行重构实验：

采用混合数字图片的数据集进行重构，如下图，对一张两数字重叠的图片进行重构，重构后的数字用不同颜色显示在同一张图上。$L:(l_1,l_2)$表示两数字的真实标签，$R:(r_1,r_2)$表示预测出并重构的两数字，带$*$标识的那两列表示重构的数字既不是标签数字也不是预测出的数字（即挑取其他capsule进行重构）。
![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/Dynamic_Routing_11.png)

---
## 第二篇《Matrix Capsules with EM routing》的解析

[Matrix Capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)

紧接着上一篇，Hinton以第一作者发布的了这篇论文，现已被ICLR接收。那这一篇与前一篇动态路由Capsule有什么不同呢？

论文中提出三点：

- (1). 前一篇采用capsule输出向量(pose vector)的模长作为实体出现的概率，为了保证模长小于1，采用了无原则性质的非线性操作，从而让那些活泼的capsule在路由迭代中被打压。
- (2).计算两个capsule的一致性（相似度）时，前一篇用两向量间夹角的余弦值来衡量（即指相似程度的迭代：$b_{ij}=b_{ij}+\hat{u}_{j|i} v_j$）。与混合高斯的对数方差不同，余弦不能很好区分"好的一致性"和"非常好的一致性"。
- (3).前一篇采用的pose是长度为$n$的向量，变换矩阵$W_{ij}$具有$n_i*n_j$个参数（如$W_{ij}=[8*16], n_i=8,n_j=16$）。 而本文采用的带$n$个元素的矩阵作为pose，这样变换矩阵$W_{ij}$具有$n$个参数（如$n=4*4)$。

从而Hinton设计了新的Capsule的结构：

|结构|图示|
|-|-|
|a). 4*4的pose矩阵,表示该capsule代表的实体,对应第(3)点.<br><br>b). 1个激活值,表示该capsule代表实体出现的概率,对应第(1)点.|![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/Capsules_matrix_7.png)|

且上一篇采用的是动态路由的方法完成了Capsule网络，而这儿将则采用EM路由算法。这也对应着上述第（2）、（3）点。

### 理解EM Routing前的准备
我们先定义下标符号：

- $i$: 指$L$层中的某个capsule。
- $c$: 指$L+1$层中的某个capsule。
- $h$: 指Pose矩阵中的某维，共16维。

为了更好理解EM-Routing算法的过程，我们先理理前期思路：

与前一篇方法类似，$L$层的$Cpasule_i$的输出($pose$)需要经过矩阵变换$W_{ic}=[4*4]$，得到它对$L+1$层的$Capsule_c$的$pose$的投票$V_{ic}$(维度和$pose$一样都是$[4*4]$)之后，才能进入Routing更新。而当$h$为$4*4$中的某一维时，让$V_{ich} = pose_{ih} * W_{ich}$，这样就可以得到$V_{ic}$了，这也对应着上述(3)。

换种解释说：$L$层的$Capsule_i$要投票给$L+1$层的$Capsule_c$，但是不同的$Capsule_c$可能需要不同变化的$Capsule_i$。所以对于每个$Capsule_c$，$Capsule_i$都有一个转换矩阵$W_{ic}$，$Capsule_i$转换后的$V_{ic}$就称投票值，而$V_{ich}$是指在$V_{ic}$在$h$维（一共4*4=16维）上的值。且变换矩阵$W_{ic}$是反向传播更新的。

文中对每个Capsule的pose建立混合高斯模型，让pose的每一维都为一个单高斯分布。即$Capsule_c$的pose中的$h$维为一个单高斯模型，$P_{ich}$是指$pose_{ch}$的值为$V_{ich}$的概率。
令$\mu_{ch}$和$\sigma_{ch}^2$分别为$Capsule_c$在$h$维上的均值和方差，则：
$$
P_{ich}=\frac{1}{\sqrt{2 \pi \sigma_{ch}^2}}{exp\big({-\frac{(V_{ich}-\mu_{ch})^2}{2\sigma_{ch}^2}}}\big)
$$
$$
ln(P_{ich})=  -\frac{(V_{ich}-\mu_{ch})^2}{2\sigma_{ch}^2} - ln(\sigma_{ch})-\frac{ln(2\pi)}{2}
$$

再令$r_i$为$Capsule_i$的激活值，就可以写出该$Capsule_c$在$h$维度上的损失$cost_{ch}$:
$$
cost_{ch}=\sum_i -r_i ln(P_{ich})=\frac{\sum_i{r_i\sigma_{ch}^2}}{2\sigma_{ch}^2} + \big(ln(\sigma_{ch})+ \frac{ln(2\pi)}{2}\big)\sum_i r_i=\big(ln(\sigma_{ch}+k)\big)\sum_i r_i
$$
值得注意的是，这里的$\sum_i$ 并不是指$L$层的所有$capsule_i$，而是指可以投票给$Capsule_c$的$Capsule_i$，这点之后会解释。 另外式子中的$k$是一个常数，它可以通过反向传播进行更新，这个在后面也会提到。

$Capsule_c$的激活值可用下面公式得出：
$$
a_c=logistic\big(\lambda\big(b_c-\sum_h cost_{ch}\big)\big)
$$
其中$-b_c$代表$Capsule_c$在$h$维上的代价均值，它可以通过反向传播进行更新。而$\lambda$是温度倒数，是超参，我们可以在训练过程中逐步改变它的值。文章中的logistic函数采用sigmoid函数。

好的，我们现在整理下在EM Routing要用的参数：

|参数|描述|
|-|-|
|$Capsule_i$、$Capsule_c$|分别指为$L$层、$L+1$层的某个$Capsule$|
|$W_{ic}$|$Capsule_i$投票给$capsule_c$前的变换矩阵，通过反向传播进行更新|
|$V_{ic}$|$Capsule_i$经过矩阵变换后准备给$capsule_c$的投票值|
|$P_{ic}$|$Capsule_i$的投票值在$Capsule_c$中的混合高斯概率|
|$cost_{ch}$|$Capsule_c$在$h$维度上的损失|
|$\mu_{ch}$、$\sigma_{ch}^2$|Capsulec在h维上的均值和方差|
|$k$|$cost_{ch}$式子中的一个常数，通过反向传播进行更新|
|$a_c$|$Capsule_c$的激活值|
|$-b_c$|$Capsule_c$在$h$维上的代价均值，通过反向传播进行更新|
|$\lambda$|温度倒数，是超参，在迭代过程中逐步增大它的值|

###　EM Routing的流程：

#### 总流程：

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/EM_ROUTING_1.png)

- 梳理符号：
    + $R_{ic}$表示$Capsule_i$给$Capsule_c$投票的加权系数
    + $M_c$、$S_c$ 表示$Capsule_c$的Pose的期望和方差
    + $size(L+1)$ 表示$Capsule_i$要投票给$L+1$层的Capsule的总数，即与$Capsule_i$有关系的$Capsule_c$的总数
- 传入$L$层的所有$Capsule_i$的激活值$a$和矩阵转换后的投票值$V$，输出$L+1$层所有$Capsule_c$的激活值$a^{'}$和Pose最终的期望值。
- 对投票加权系数初始化后就进行EM算法迭代（一般迭代3次）：
    + 对每个$Capsule_c$，M-step得到它的Pose的期望和方差
    + 对每个$Capsule_i$，E-step中得到它更新后的对所有$Capsule_c$投票的加权系数。
- 再次提醒，这里$Capsule_i$不是投票给所有而是部分$Capsule_c$。此处的”所有“是为了表述方便。具体理由之后会解释。

#### 分析M-Step步骤:

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/EM_ROUTING_2.png)

- 梳理符号：
    + $cost_h$即为$cost_{ch}$，是$Capsule_c$在$h$维度上的损失.
    + $\beta_v$即为$cost_{ch}$式子中的一个常数k（前面提及过），通过反向传播进行更新.
    + $\beta_a$即为$Capsule_c$在$h$维上的代价均值$-b_c$的负数(前面提及过)
    + $\lambda$即为温度倒数。是超参，在迭代中将逐步增大它的值（前面提及过），通过反向传播进行更新
- 对于某$Capsule_c$，传入所有$Capsule_i$对它的投票加权系数$R_{:c}$、所有$Capsule_i$的激活值$a$、所有$Capsule_i$矩阵转换后对它的投票值$V_{:c:}$，输出该$Capsule_c$的激活值以及pose期望方差。

#### 分析E-Step步骤:

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/EM_ROUTING_3.png)

- 梳理符号：
    + $p_c$即为$P_{ic}$，是$Capsule_i$的投票值在$Capsule_c$中的混合高斯概率，前面提及过。
    + $r$即为$r_i$， 是$Capsule_i$给所有$Capsule_c$投票的加权系数
- 对于某$Capsule_i$， 传入它对所有$Capsule_c$的投票$V_i$、所有$Capsule_c$的激活值以及pose的均值和方差，得到它更新后的对所有$Capsule_c$投票的加权系数$r_i$。


经过一轮的公式符号轰炸，我们就明白了EM-Routing的整体流程。


### Matrix Capsules 网络模型

接下来我们要看下Hinton设计的Matrix Capsule的网络模型：

![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/EM_ROUTING_4.png)

由于涉及到Capsule的卷积操作，此处先定义一些概念，以ConvCaps1层为例子，在ConvCaps1层中：

- 含有C个channel，每个channel含有6*6=36个capsule。
- 不同的channel含有不同的类型capsule，同一channel的capsule的类型相同但位置不同。
- 任一个capsule均有6*6-1=35个相同类型的capsule，均有C-1个位置相同的capsule。


#### 第一层ReLU Conv1的获取：

- 普通的卷积层，使用A个5×5的卷积核、步幅为2、ReLU作为激活函数，得到A×14×14的张量输出。

#### 第二层PrimaryCaps的获取:

- 从普通卷积层构建成Capsule结构的PrimaryCaps层，用了A×B×(16+1)个1×1的卷积核，步幅为1，得到B×14×14个pose和B*14*14个激活值，即有B*14*14个capsule。

#### 第三层ConvCaps1的获取：

- 此处将PrimaryCaps层（即$L$层）中的$Capsule_i$的pose进行矩阵转换后，得到对应的投票矩阵$V$。将$V$和全部$Capsule_i$的激活值传入RM-Routing中，即可得到C*6*6个$Capsule_c$的pose及激活值。 
- 此处是Capsule卷积操作，卷积核大小为K，步幅为2。
    + 对于$L+1$层的某个$Capsule_c$，需要得到投票给它的那些$Capsule_i$的投票矩阵$V_{:c}$
        * 在$L$层只有K×K*B个$Capsule_i$投票给该$Capsule_c$，对这K×K*B个$Capsule_i$的pose分别进行矩阵转换，即可得到投票矩阵$V_{ic}$
        * 这里有K*K*B*C个转换矩阵$W_{ic}$，每个$W_{ic}$是4*4的矩阵（或说16维的向量）。
    + 这两层间转换矩阵$W_{ic}$共有：(K*K*B)*C*6*6*个 （而不是14*14*B*C*6*6).
    + 与普通二维卷积一样，不过卷积的乘法操作改为EM-Routing。即被卷积的是$L$层的所有$Capsule_i$的投票矩阵$V_i$和激活值，卷积结果是C*6*6个$Capsule_c$的pose及激活值。
        * 每个$L+1$层的$Caspule_c$都采用capsule卷积（EM-Routing）对应$L$层的K*K*B个$Capsule_i$，从而得到该$Caspule_c$的pose和激活值。
        * 对于$L$层的中心位置的B个$Capsule_i$，它们每个$Capsule_i$，都只投票给卷积时卷积核滑过它们的对应的$Capsule_c$（共K*K*C个）。而$L$层的边缘位置的每个$Capsule_i$投票给$L+1$层的$Capsule_c$个数将小于K*K*C个。如$L$层最左上位置的B个$Capsule_i$，它们只能投给$L+1$层最左上角的C个$Capsule_c$（只有$L+1$层的这个位置执行卷积时候卷积核才滑过$L$层最左上角）

#### 第四层ConvCaps2的获取：

- capsule卷积层，与ConvCaps1一样的操作。
- 采用的卷积核K=3，步幅=1 。得到D*4*4个capsule的pose与激活值。 

#### 第五层Class Capsules的获取：

- 从capsule卷积层到最后一层的操作，和前面的做法不同。
    + 相同类型（即同一channels）的capsule共享转换矩阵，所以两层间共有D*E个转换矩阵$W_j$，每个$W_j$是4*4的矩阵（或说16维的向量）。
    + 拉伸后的扁长层（Class Capsules层）无法表达capsule的位置信息，而前面ConvCaps2层的每个$capsule_i$都有对应的位置信息。为了防止位置信息的丢失，作者将每个$Capsule_i$的位置信息（即坐标）分别加到它们的投票矩阵$V_ij$的一二维上。随着训练学习，共享转换矩阵$W_j$能将$V_ij$的一二维与$Capsule_i$的位置信息联系起来，从而让Class Capsules层的$Capsule_j$的pose的一二维携带位置信息。
- 得到E个capsule， 每个capsule的pose表示对应calss实体，而激活值表示该实体存在的概率。
- 这样就可以单独拿出capsule的激活值做概率预测，拿capsule的pose做类别实体重构了。

整体的Matrix Capsules网络模型就梳理完成了。现在还剩下损失函数了。

### 损失函数：传播损失(Spread Loss):

为了使训练过程对模型的初始化以及超参的设定没那么敏感，文中采用传播损失函数来最大化被激活的目标类与被激活的非目标类之间的概率差距。a_t表示target的激活值，a_i表示Class_Capsules中除t外第i个的激活值：
$$
L_i=\big(\max \big(0,m-(a_t-a_i)\big)\big)^2 , \quad  L=\sum_{i\neq t} L_i
$$

m将从0.2的小幅度开始，在训练期间将其线性增加到0.9，避免无用胶囊的存在。那为什么要这样做呢？ 
小编的理解是：

 - 当模型训练得挺好时候（中后期），每个激活值a_i的比较小而a_t比较大。此时m需要设置为接近1的数值。设置m为0.9而不是1，这是为了防止分类器过度自信，是要让分类器更专注整体的分类误差。
 - 当模型初步训练时候，很多capsules起的作用不大，最终的激活值a_i和a_t相差不大，若此时m采用较大值如0.9,就会掩盖了(a_t-a_i)在参数更新的作用，而让m主导了参数更新。
     + 比如两份参数W1和W2对同样的样本得到的$a_t-a_i$值有：$W1_{a_t-a_i} < W2_{a_t-a_i}$ ，那显然W2参数优于W1参数，即W1参数应该得到较大幅度的更新。但由于处于模型初步阶段，$W_{a_t-a_i}$值很小，若此时m较大，则m值主导了整体loss。换句话说，m太大会导致W1和W2参数更新的幅度相近，因为$a_t-a_i$被忽略了。
     + 不过参数的更新幅度取决于对应的导数，由于此处的spread loss含有平方，所以m值的设置会关系到参数的导数，从而影响到参数更新的幅度 （有些loss由于公式设计问题会导致从loss看不出参数更新的幅度，如若此处将Spread loss的平方去掉，参数的更新就和m无关了）。

---

### 实验结果

作者采用smallNORB数据集（具有5类玩具的灰度立体图片：飞机、汽车、卡车、人类和动物）上进行了实验。选择smallNORB作为capsule网络的基准，因为它经过精心设计，是一种纯粹的形状识别任务，不受上下文和颜色的混淆，但比MNIST更接近自然图像。下图为在不同视角上的smallNORB物体图像：
![](https://github.com/luonango/luonango.github.io/raw/master/img/pictures/EM_ROUTING_6.png)

该smallNORB数据集，CNN中的基准线是：测试集错误率5.2%，参数总量4.2M。而作者使用小型的capsule网络（A=64, B = 8, C = D = 16，参数总量68K），达到2.2%的测试集错误率，这也击败了当前最好的记录。 

作者还实验了其他数据集。采用上图的Matrix Capsules网络及参数，在MNIST上就达到了0.44％的测试集错误率。如果让A=256，那在Cifar10上的测试集错误率将达到11.9％.

文章后面还讨论了在对抗样本上，capsule模型和传统卷积模型的性能。实验发现，在白箱对抗攻击时，capsule模型比传统的卷积模型更能抵御攻击。而在黑箱对抗攻击时，两种模型差别不大。感兴趣的话可以看看论文中对这部分实验的设置及分析。

---

## 参考链接：

- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
- [Matrix Capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)
- [浅析 Hinton 最近提出的 Capsule 计划](https://zhuanlan.zhihu.com/p/29435406)
- [Capsule Networks Explained](https://kndrck.co/posts/capsule_networks_explained/)
- [先读懂CapsNet架构然后用TensorFlow实现：全面解析Hinton的提出的Capsule](https://www.jiqizhixin.com/articles/2017-11-05)
- [Capsule官方代码开源之后，机器之心做了份核心代码解读](https://www.jiqizhixin.com/articles/capsule-implement-sara-sabour-Feb02)
- [CapsulesNet 的解析及整理](https://zhuanlan.zhihu.com/p/30970675)
- [Understanding Matrix capsules with EM Routing (Based on Hinton's Capsule Networks)](https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/)
- [Dynamic Routing官方源代码-Tensorflow](https://github.com/Sarasra/models/tree/master/research/capsules) 
- [Dynamic Routing:HuadongLiao的CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)
- [Dynamic Routing:Xifeng Guo的CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)

