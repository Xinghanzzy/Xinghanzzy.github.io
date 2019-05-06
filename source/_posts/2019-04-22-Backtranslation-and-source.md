---
title:  Back-Translation和源语端数据增强
date: 2019-04-22 10:37:51
header-img: /images/post-bg-coffee.jpeg
tags:
    - NMT
    - NLP

---

# Back-Translation



# 源语端数据增强

这是自动化所张家俊老师2016年的ACL文章《Exploiting Source-side Monolingual Data in Neural Machine Translation》

## 摘要

我们两种方法来充分利用NMT中的源边单语数据。 

1. 第一种方法采用自学习算法生成用于NMT训练的合成大规模并行数据。 
2. 第二种方法应用多任务学习框架，使用两个NMT同时预测翻译和重新排序的源侧单语句。

都是利用单语数据**提高Encoder**性能的方法

选用**50%**的单词在词表中的源语单语数据比较好（论文中未指出数据量的影响）

## 方法

### 自学习算法

使用可用的对齐句子对构建基线机器翻译系统，然后通过将源侧单语句与基线系统进行翻译来获得更多的合成并行数据。

合成目标部分可能对NMT的解码器模型产生负面影响。为了解决这个问题，我们可以通过**冻结**合成数据的**解码器网络参数**来区分NMT训练期间合成双语句子的原始双字。

自学习算法可以改进NMT的Encoder。

**有效性推测：**

​	Encoder见到更多的情况，提取信息能力变强

> 源侧单语数据在词汇表中提供了更多的单词排列。 我们的RNN编码器网络模型将进行优化，以便很好地解释所有单词排列。 
>
> the source-side monolingual data provides much more permutations of words in the vocabulary. Our RNN encoder network model will be optimized to well explain all of the word permutations. 

### 多任务学习

应用多任务学习框架来同时预测目标翻译和重新排序的源侧句子。背后的主要思想是我们构建了两个NMT：一个是在对齐的句子对上训练来预测来自源句子的目标句子，而另一个是在源侧单语语料库上训练来预测来自原始来源的重新排序的源句子sentences1。应当注意，两个NMT共享相同的编码器网络，以便它们可以相互帮助以加强编码器模型。

![](https://ws1.sinaimg.cn/large/4ac7f217ly1g2bb0yhdavj20fi0eh0tk.jpg)

- machine translation task

  input：src

  lable：tgt

- sentence reordering task

  input：src

  lable：src_reorder

  pre-ordering rules proposed by (Wang et al., 2007),

  一个将src按照tgt语言语序排列的工具(很慢)

  > which can permutate the words of the source sentence so as to approximate the target language word order.
  >
  > 它可以排列源句的单词，以便近似目标语言单词顺序。

![](https://ws1.sinaimg.cn/large/4ac7f217ly1g2bby7r8bbj20fg05h0t2.jpg)

- 训练

  **1**epoch sentence reordering task + **4** epoch machine translation task



## 结果

![](https://ws1.sinaimg.cn/large/4ac7f217ly1g2bc288yyaj20wu0gbdk5.jpg)























