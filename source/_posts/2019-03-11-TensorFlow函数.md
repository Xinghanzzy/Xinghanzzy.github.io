---
layout:     post   				    # 使用的布局（不需要改）
title:      TensorFlow函数 				# 标题 
subtitle:   TensorFlow常用函数 #副标题
date:       2019-03-11 				# 时间
author:     XH 						# 作者
header-img: /images/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - NMT
    - TensorFlow
    - NLP

---


> [我的的博客](https://xinghanzzy.github.io/)



# TensorFlow函数

## tf.tile

```python 
# key_masks：(N, T_k) N:batch_size T:maxlen
tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k) 
```

在input的每一维 复制对应的次数

```python
tile([x, y], [a, b])
# x 复制 a 次
# y 复制 b 次
```



## tf.reduce_sum

```python 
tf.reduce_sum(keys, axis=-1)
# input按照第n维加和 
# 感觉就是那一维度没了
# [N, T, num_units] -> [N, T]

# 如果没有第二个参数，就是对矩阵中所有元素求和
def reduce_sum(input_tensor,
               axis=None,
               keepdims=None,
               name=None,
               reduction_indices=None,
               keep_dims=None):
  """Computes the sum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.
"""
```

## tf.arg_max

tf.argmax() 与 numpy.argmax() 方法的意思是一致的， 即：

    axis = 0 时       返回每一列最大值的位置索引
    
    axis = 1 时       返回每一行最大值的位置索引

##tf.nn.embedding_lookup

tf.nn.embedding_lookup()就是根据input_ids中的id，寻找embeddings中的第id行。比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回。

embedding_lookup不是简单的查表，id对应的向量是可以训练的，训练参数个数应该是 category num*embedding size，也就是说lookup是一种全连接层。

我的理解是表中数是参与训练的

> 作者：大师鲁 
> 来源：CSDN 
> 原文：https://blog.csdn.net/laolu1573/article/details/77170407 
> 版权声明：本文为博主原创文章，转载请附上博文链接！