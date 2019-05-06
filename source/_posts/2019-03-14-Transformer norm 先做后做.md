---
layout:     post
title:      Transformer norm 先做后做
subtitle:   Transformer layer normalization 先做后做
date:       2019-03-14
author:     XH
header-img: /images/post-bg-debug.png
catalog: true
tags:
    - Transformer
    - NMT
---


>并不适合阅读的个人文档。

# Normalization  先做 和 后做 

![](https://ws1.sinaimg.cn/large/4ac7f217ly1g12hedto9jj20d208e3yt.jpg)

- 后做是传统做法

  后做是 input x ,  x residual 到后面, x 进行function(multi-head attention),  f(x) 进行dropout，df(x) + x  （residual）, layer normalization(df(x) + x  )

  **x’->LN(df(x) + x)**

- 先做是当前性能好的做法

  后做是 input x ,  x residual 到后面, x 进行LN，然后进行function(multi-head attention), 最后进行dropout在和残差相加，dfLN(x) + x  （residual）

  **x‘->x + (df(LN(x))**

  > 最顶层+LN

  > 反向传播更好

- 思考

  为何不使用 xl(residual) -> F -> LN -> add(residual) ->xl+1
  $$
  x_l(residual) -> F -> LN -> add(residual) ->x_{l+1}
  $$
  因为这样的操作，送入F的x是在加和后没进行norm的，容易出现梯度消失/梯度爆炸问题
