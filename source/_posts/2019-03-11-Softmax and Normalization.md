---
layout:     post   				    # 使用的布局（不需要改）
title:      Softmax and Normalization # 标题 
subtitle:   Softmax and Normalization理解 #副标题
date:       2019-03-11 				# 时间
author:     XH 						# 作者
header-img: /images/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - NMT
    - TensorFlow
    - NLP

---


> [三分钟带你对 Softmax 划重点](https://blog.csdn.net/red_stone1/article/details/80687921/)
>
> [深度学习中 Batch Normalization为什么效果好？-知乎](https://www.zhihu.com/question/38102762/answer/85238569?utm_source=wechat_session&utm_medium=social&utm_oi=629421655403925504/ )
>
> [详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246/)
>
> [我的的博客](https://xinghanzzy.github.io/)



# Softmax

在数学，尤其是概率论和相关领域中，Softmax函数，或称归一化 指数函数，是逻辑函数的一种推广。

Softmax函数实际上是有限项离散概率分布的梯度对数归一化。

它能将一个含任意实数的K维向量  “压缩”到另一个K维实向量  中，使得每一个元素的范围都在$(0, 1)$之间，并且所有元素的和为$1$。
$$
S_i=\frac {e^{V_i}} {\sum_{i}^{c} {e^{V_i}}}
$$

在实际应用中遇到提前$padding$时候将$V_i$设置为$-2^{32}+1$这种情况

为了是经过$Softmax$之后可以把值变为0



# Normalization

## Batch Normalization

### what

​	顾名思义，batch normalization嘛，就是“批规范化”咯。Google在ICML文中描述的非常清晰，即在每次SGD时，通过mini-batch来对相应的activation做规范化操作，**使得结果（输出信号各个维度）的均值为0，方差为1.** 

​	最后的“scale and shift”操作则是为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入（即当![\gamma^{(k)}=\sqrt{Var[x^{(k)}]}, \beta^{(k)}=E[x^{(k)}]](https://www.zhihu.com/equation?tex=%5Cgamma%5E%7B%28k%29%7D%3D%5Csqrt%7BVar%5Bx%5E%7B%28k%29%7D%5D%7D%2C+%5Cbeta%5E%7B%28k%29%7D%3DE%5Bx%5E%7B%28k%29%7D%5D)   $\gamma^{(k)}=\sqrt{Var[x^{(k)}]}, \beta^{(k)}=E[x^{(k)}]$ 时候，相当于未做normalization），从而保证整个network的capacity（容量）。（有关capacity的解释：实际上BN可以看作是在原模型上加入的“新操作”，这个新操作很大可能会改变某层原来的输入。当然也可能不改变，不改变的时候就是“还原原来输入”。如此一来，既可以改变同时也可以保持原输入，那么模型的容纳能力（capacity）就提升了。）

![preview](https://pic2.zhimg.com/9ad70be49c408d464c71b8e9a006d141_r.jpg) 

### where

​	BN可以应用于网络中任意的activation set。文中还特别指出在CNN中，BN应作用在非线性映射前，即对做规范化。 

### why

​	说到底，BN的提出还是为了克服深度神经网络难以训练的弊病。其实BN背后的insight非常简单，只是在文章中被Google复杂化了。

> 首先来说说“Internal Covariate Shift”。文章的title除了BN这样一个关键词，还有一个便是“ICS”。
>
> 大家都知道在统计机器学习中的一个经典假设是“源空间（source domain）和目标空间（target domain）的数据分布（distribution）是一致的”。如果不一致，那么就出现了新的机器学习问题，如，transfer learning/domain adaptation等。
>
> covariate shift就是分布不一致假设之下的一个分支问题，它是指源空间和目标空间的条件概率是一致的，但是其边缘概率不同，即：对所有![x\in \mathcal{X}](https://www.zhihu.com/equation?tex=x%5Cin+%5Cmathcal%7BX%7D),![P_s(Y|X=x)=P_t(Y|X=x)](https://www.zhihu.com/equation?tex=P_s%28Y%7CX%3Dx%29%3DP_t%28Y%7CX%3Dx%29)，但是![P_s(X)\ne P_t(X)](https://www.zhihu.com/equation?tex=P_s%28X%29%5Cne+P_t%28X%29). 大家细想便会发现，的确，对于神经网络的各层输出，由于它们经过了层内操作作用，其分布显然与各层对应的输入信号分布不同，而且差异会随着网络深度增大而增大，可是它们所能“指示”的样本标记（label）仍然是不变的，这便符合了covariate shift的定义。由于是对层间信号的分析，也即是“internal”的来由。

那BN到底是什么原理呢？说到底还是**为了防止“梯度弥散”**。关于梯度弥散，大家都知道一个简单的栗子：![0.9^{30}\approx 0.04](https://www.zhihu.com/equation?tex=0.9%5E%7B30%7D%5Capprox+0.04)。在BN中，是通过将activation规范为均值和方差一致的手段使得原本会减小的activation的scale变大。可以说是一种更有效的local response normalization方法（见4.2.1节）。



梯度弥散、梯度爆炸

> 靠近输出层的hidden layer 梯度大，参数更新快，所以很快就会收敛；
>
> 而靠近输入层的hidden layer 梯度小，参数更新慢，几乎就和初始状态一样，随机分布。
>
> 在上面的四层隐藏层网络结构中，第一层比第四层慢了接近100倍！！
>
> 这种现象就是梯度弥散（vanishing gradient problem）。而在另一种情况中，前面layer的梯度通过训练变大，而后面layer的梯度指数级增大，这种现象又叫做梯度爆炸(exploding gradient problem)。
>
> 总的来说，就是在这个深度网络中，梯度相当不稳定(unstable)。


$$
\gamma^{(k)}=\sqrt{Var[x^{(k)}]}, \beta^{(k)}=E[x^{(k)}]
$$

**Batch Normalization —— 纵向规范化** 

**Layer Normalization —— 横向规范化** 



> 原来的网络，如果简单地增加深度，会导致梯度弥散或梯度爆炸。对于该问题的解决方法是正则化初始化和中间的正则化层（Batch Normalization），这样的话可以训练几十层的网络。
>
> 虽然通过上述方法能够训练了，但是又会出现另一个问题，就是退化问题，网络层数增加，但是在训练集上的准确率却饱和甚至下降了。这个不能解释为overfitting，因为overfit应该表现为在训练集上表现更好才对。退化问题说明了深度网络不能很简单地被很好地优化。
>
> residual