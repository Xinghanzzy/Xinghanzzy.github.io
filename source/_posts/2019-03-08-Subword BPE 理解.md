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
> 感谢[夏天的米米阳光CSDN](http://blog.csdn.net/u013453936/article/details/80878412(http://www.runoob.com/w3cnote/git-guide.html)
>
> 感谢[subword-units](http://plmsmile.github.io/2017/10/19/subword-units/)
>
> [我的的博客](https://xinghanzzy.github.io/)



# Subword



## learn BPE

BPE词表学习，首先统计词表词频，然后每个单词表示为一个字符序列，并加上一个特殊的词尾标记 

>  apple eat 这两个的e是不同的
>
> appale	e<\w>
>
> eat		e

取出频率最高的‘a b’加入词表中，并将‘a b’替换为‘ab’,重复过程

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



