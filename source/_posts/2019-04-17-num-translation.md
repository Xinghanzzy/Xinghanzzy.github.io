---
title: num translation
date: 2019-04-17 20:27:25
header-img: /images/ind123123ex.jpg
catalog: true
mathjax: true
tags:
    - 深度学习
    - NMT

---

> 施工中



## 对齐关系

对齐关系的获取是这里面的难点

我们的翻译模型用的是谷歌的Transformer模型

>  tensor2tensor==1.0.14 
>
> tensorflow==1.4.0
>
> CUDA 8.0 Python3

Transformer模型最大的特点是对注意力机制$Attention$的大量使用

而$Encoder-Decoder\ Attention$是用$Encoder$层的输出以及$Decoder$上一层的输出作为输入进行计算

即：
$$
Query:Encoder\_Output
$$

$$
Key=Value:Decoder(上一层输出)
$$

在这个计算中，Q K相乘获得的矩阵隐性包含着我们需要的对齐信息。

Q K 相乘的矩阵是 （h*batch_size,len_q,len_k）的

我们对最后一层的QK矩阵多头取平均，就得到了我想要的结果