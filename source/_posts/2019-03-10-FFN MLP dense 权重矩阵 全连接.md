---
layout:     post   				    # 使用的布局（不需要改）
title:      FFN MLP dense 权重矩阵 全连接 # 标题 
subtitle:   FFN MLP dense 权重矩阵 全连接 区别理解 #副标题
date:       2019-03-11 				# 时间
author:     XH 						# 作者
header-img: /images/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - NMT
    - TensorFlow
    - NLP

---


> 参考文章，感谢作者付出。
>
> [直观理解神经网络最后一层全连接+Softmax](https://blog.csdn.net/blogshinelee/article/details/84826837/)
>
> [花书+吴恩达深度学习（一）前馈神经网络（多层感知机 MLP）](https://blog.csdn.net/zhq9695/article/details/84337984/)
>
> [如何理解softmax ](https://blog.csdn.net/qq_31713935/article/details/78784408/)
>
> [【AI数学】Batch-Normalization详细解析](https://blog.csdn.net/leviopku/article/details/83109422/)
>
> [我的的博客](https://xinghanzzy.github.io/)



> 一句话：FNN=MLP=n*dense=n*relu(Wx+b) 

# FeedForward

FNN FFN？傻傻分不清楚 

**前馈神经网络**（feedforward neural network），又称作**深度前馈网络**（deep feedforward network）、**多层感知机**（multilayer perceptron，MLP） 

每一个神经元由**一个线性拟合**和**一个非线性激活函数**组成 

不同层之间就是全连接

**我们的任务就是找到权值和偏置这些参数的值，使得输出的东西让我们满意，达到我们的要求。** 

我的理解是 输入经过一层（放大），然后在经过一层（缩小）

层的选择 $dense $还有 $conv1d$ 都有，和同学讨论，谷歌的transformer实现用的**卷积**，听说是卷积快一些。

# MLP

多层感知器（MLP，Multilayer Perceptron）是一种前馈人工神经网络模型 

也叫人工神经网络（ANN，Artificial Neural Network） 

除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构 

![](https://ws1.sinaimg.cn/large/4ac7f217ly1g0zzty4jbuj207x03mwer.jpg)

多层感知机层与层之间是全连接的（全连接的意思就是：上一层的任何一个神经元与下一层的所有神经元都有连接） 



# 全连接层 dense

全连接层的每一个结点都与上一层的所有结点相连，用来把前边提取到的特征综合起来。由于其全相连的特性，一般全连接层的参数也是最多的。 

**全连接层**将**权重矩阵**与输入向量相乘再加上偏置 

![一维全连接](https://ws1.sinaimg.cn/large/4ac7f217ly1g1001xjzuqj20b00bb74v.jpg)

​										  一维全连接

![二维全连接层 ](https://ws1.sinaimg.cn/large/4ac7f217ly1g10023khdej20dz0dpwfd.jpg)

​										二维全连接层

```python
dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)
# inputs: 输入数据，2维tensor.
# units: 该层的神经单元结点数。
# activation: 激活函数.
# use_bias: Boolean型，是否使用偏置项.
# kernel_initializer: 卷积核的初始化器.
# bias_initializer: 偏置项的初始化器，默认初始化为0.
# kernel_regularizer: 卷积核化的正则化，可选.
# bias_regularizer: 偏置项的正则化，可选.
# activity_regularizer: 输出的正则化函数.
# trainable: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
# name: 层的名字.
# reuse: Boolean型, 是否重复使用参数.
# 全连接层执行操作 outputs = activation(inputs.kernel + bias)

# 如果执行结果不想进行激活操作，则设置activation=None。
```

