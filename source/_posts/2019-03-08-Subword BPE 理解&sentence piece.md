---
layout:     post   				    # 使用的布局（不需要改）
title:      Subword BPE 理解 				# 标题 
subtitle:   Subword 学习记录 #副标题
date:       2019-03-08 				# 时间
author:     XH 						# 作者
header-img: /images/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - Subword
    - BPE
---


> 正所谓前人栽树，后人乘凉。
>
> 感谢[夏天的米米阳光CSDN](http://blog.csdn.net/u013453936/article/details/80878412)
>
> 感谢[自然语言处理之_SentencePiece分词](https://www.jianshu.com/p/9b93cef1ca56 )
>
> 感谢[subword-units](http://plmsmile.github.io/2017/10/19/subword-units/)
>
> [我的的博客](https://xinghanzzy.github.io/)



# Subword

**BPE的训练和解码范围都是一个词的范围。** 



## learn BPE

BPE词表学习，首先统计词表词频，然后每个单词表示为一个字符序列，并加上一个特殊的词尾标记 

>  apple eat 这两个的e是不同的
>
> appale	e<\w>
>
> eat		e

取出频率最高的‘a b’加入词表中，并将‘a b’替换为‘ab’,重复过程

codec文件中保存的就是训练过程的字符对，文件中最开始的是训练时最先保存的字符，即具有较高的优先级。 

### 例子

the、and、$date

```
#version: 0.2 
t h
i n
a n
th e</w>
t i
r e
e n
an d</w>

d ate</w>
it s</w>
er e</w>
t a
o g
d s</w>
ent s</w>
ro m</w>
f rom</w>
ig h
committe e</w>
on e</w>
st ate</w>
i r</w>
the ir</w>
a y</w>
$ date</w>
```

## apply bpe

按在词的范围中进行编码的，首先将词拆成一个一个的字符，然后按照训练得到的codec文件中的字符对来合并。 



## 论文代码简单实现

(subword-units 实现)

```python
import re

def process_raw_words(words, endtag='-'):
    '''把单词分割成最小的符号，并且加上结尾符号'''
    vocabs = {}
    for word, count in words.items():
        # 加上空格
        word = re.sub(r'([a-zA-Z])', r' \1', word)
        word += ' ' + endtag
        vocabs[word] = count
    return vocabs

def get_symbol_pairs(vocabs):
    ''' 获得词汇中所有的字符pair，连续长度为2，并统计出现次数
    Args:
        vocabs: 单词dict，(word, count)单词的出现次数。单词已经分割为最小的字符
    Returns:
        pairs: ((符号1, 符号2), count)
    '''
    #pairs = collections.defaultdict(int)
    pairs = dict()
    for word, freq in vocabs.items():
        # 单词里的符号
        symbols = word.split()
        for i in range(len(symbols) - 1):
            p = (symbols[i], symbols[i + 1])
            pairs[p] = pairs.get(p, 0) + freq
    return pairs

def merge_symbols(symbol_pair, vocabs):
    '''把vocabs中的所有单词中的'a b'字符串用'ab'替换
    Args:
        symbol_pair: (a, b) 两个符号
        vocabs: 用subword(symbol)表示的单词，(word, count)。其中word使用subword空格分割
    Returns:
        vocabs_new: 替换'a b'为'ab'的新词汇表
    '''
    vocabs_new = {}
    raw = ' '.join(symbol_pair)
    merged = ''.join(symbol_pair)
    # 非字母和数字字符做转义
    bigram =  re.escape(raw)
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word, count in vocabs.items():
        word_new = p.sub(merged, word)
        vocabs_new[word_new] = count
    return vocabs_new

raw_words = {"low":5, "lower":2, "newest":6, "widest":3}
vocabs = process_raw_words(raw_words)
# print(vocabs)

num_merges = 10
print (vocabs)
for i in range(num_merges):
    pairs = get_symbol_pairs(vocabs)
    # 选择出现频率最高的pair
    symbol_pair = max(pairs, key=pairs.get)
    print(pairs)
    print(symbol_pair)
    vocabs = merge_symbols(symbol_pair, vocabs)
print (vocabs)
```

输出

```
{' l o w -': 5, ' l o w e r -': 2, ' n e w e s t -': 6, ' w i d e s t -': 3}
{('l', 'o'): 7, ('o', 'w'): 7, ('w', '-'): 5, ('w', 'e'): 8, ('e', 'r'): 2, ('r', '-'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('e', 's'): 9, ('s', 't'): 9, ('t', '-'): 9, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'e'): 3}
('e', 's')
{('l', 'o'): 7, ('o', 'w'): 7, ('w', '-'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'es'): 6, ('es', 't'): 9, ('t', '-'): 9, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'es'): 3}
('es', 't')
{('l', 'o'): 7, ('o', 'w'): 7, ('w', '-'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est'): 6, ('est', '-'): 9, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est'): 3}
('est', '-')
{('l', 'o'): 7, ('o', 'w'): 7, ('w', '-'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est-'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est-'): 3}
('l', 'o')
{('lo', 'w'): 7, ('w', '-'): 5, ('w', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est-'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est-'): 3}
('lo', 'w')
{('low', '-'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('n', 'e'): 6, ('e', 'w'): 6, ('w', 'est-'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est-'): 3}
('n', 'e')
{('low', '-'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('ne', 'w'): 6, ('w', 'est-'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est-'): 3}
('ne', 'w')
{('low', '-'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('new', 'est-'): 6, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est-'): 3}
('new', 'est-')
{('low', '-'): 5, ('low', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est-'): 3}
('low', '-')
{('low', 'e'): 2, ('e', 'r'): 2, ('r', '-'): 2, ('w', 'i'): 3, ('i', 'd'): 3, ('d', 'est-'): 3}
('w', 'i')
{' low-': 5, ' low e r -': 2, ' newest-': 6, ' wi d est-': 3}
```



# Sentence piece

## 说明

​	SentencePiece是一个google开源的自然语言处理工具包。网上是这么描述它的：数据驱动、跨语言、高性能、轻量级——面向神经网络文本生成系统的无监督文本词条化工具。 

​	一些组合常常出现，但事先并不知道，于是我们想让机器自动学习经常组合出现的短语和词。SentencePiece就是来解决这个问题的。它需要大量文本来训练。

 SentencePiece的用途不限于自然语言处理，记得DC之前有一个药物分子筛选的比赛，蛋白质的一级结构是氨基酸序列，需要研究氨基酸序列片断，片断的长度又是不固定的，此处就可以用SentencePiece进行切分。原理是重复出现次数多的片断，就认为是一个意群（词）。(**未经过验证**)

## 安装

 SentencePiece分为两部分：训练模型和使用模型，训练模型部分是用C语言实现的，可编成二进程程序执行，训练结果是生成一个model和一个词典文件。

 模型使用部分同时支持二进制程序和Python调用两种方式，训练完生成的词典数据是明文，可编辑，因此也可以用任何语言读取和使用。

#### 1) 在Ubuntu系统中安装Python支持

```shell
$ sudo pip install SentencePiece
```

#### 2) 下载源码

```shell
$ git clone https://github.com/google/sentencepiece
$ cd sentencepiece
$ ./autogen.sh
$ ./confiture; make; sudo make install # 注意需要先安装autogen,automake等编译工具
```

## 训练模型

```shell
$ spm_train --input=/tmp/a.txt --model_prefix=/tmp/test
# --input指定需要训练的文本文件，--model_prefix指定训练好的模型名，本例中生成/tmp/test.model和/tmp/test.vocab两个文件，vocab是词典信息。
$ spm_train --input=<input> --model_prefix=<model_name> --vocab_size=8000 --model_type=<type>
```

model_name为保存的模型为model_name.model,词典为model_name.vocab,词典大小可以人为设定vocab  _size.训练模型包括`unigram` (default), `bpe`, `char`, or `word`四种类型. 

## 使用模型

#### (1) 命令行调用

```shell
$ echo "食材上不会有这样的纠结" | spm_encode --model=/tmp/test.model
```

#### (2) Python程序调用

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
text = "食材上不会有这样的纠结" 

sp.Load("/tmp/test.model") 
print(sp.EncodeAsPieces(text))
```

## 使用技巧

 如果我们分析某个领域相关问题，可以用该领域的书籍和文档去训练模型。并不限于被分析的内容本身。训练数据越多，模型效果越好。更多参数及用法，请见git上的说明文件。

## 参考

#### (1) 用法示例

<https://pypi.org/project/sentencepiece/0.0.0/>

#### (2) 训练示例

<https://github.com/google/sentencepiece#train-sentencepiece-model>

