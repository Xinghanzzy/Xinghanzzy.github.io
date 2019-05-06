---
layout:     post   				    # 使用的布局（不需要改）
title:      Transformer阅读 				# 标题 
subtitle:   Transformer-master代码阅读 #副标题
date:       2019-03-11 				# 时间
author:     XH 						# 作者
header-img: /images/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
usemathjax: true
tags:								#标签
    - NMT
    - Transformer
    - TensorFlow
    - NLP

---


> [我的的博客](https://xinghanzzy.github.io/)
>
> [机器翻译模型Transformer代码详细解析](https://blog.csdn.net/mijiaoxiaosan/article/details/74909076 )
>
> [代码地址](https://github.com/Xinghanzzy/transformer-simple/)



# Transformer

动机：前几天和同学讨论一下decoder对mask是否有优化，发现自己对tensor shape记得并不好，这两天再过一次 加深一下印象并记录

代码：17年老代码，好处是写的很简单

## 数据预处理 prepro

1. 洗数据：去除非拉丁字符 
2. 词频统计，$Counter()$，降序
3. PAD,UNK, S, /S，文件开头，数量1000000000

## train

- 加载词表，$2*2$个list，$lang2idx，idx2lang$

- 构图

  - 默认图

  - 读取数据batch，输出为ont-hot idx

    > input : none
    >
    > output : (N, T) 	

    ```python
    # N：batch_size  T:maxlen 最大句长
    self.x, self.y, self.num_batch = get_batch_data() # (N, T) 		
    	lang2idx，idx2lang
    	word2idx: OOV->"<UNK>"(1) 句尾+</S>
        # x和y后面pad上x和y初始元素个数和句子最大长度差的那么多数值 0
        # pad 垫，填补
    	PAD:补全0，maxlen
        num_batch = len(X) // hp.batch_size # 整除
        # Create Queues
        input_queues = tf.train.slice_input_producer([X, Y])
        x, y = tf.train.shuffle_batch
    ```

  - decoder input : 句尾去掉\S句首+ S (2)

    > input :  (N, T) 
    >
    > output : (N, T) 
    >
    > 用self.y来初始化解码器的输入。decoder_inputs和self.y相比，去掉了最后一个句子结束符，而在每句话最前面加了一个初始化为2的id，即S
    >
    > 
    >  scr: 今天天气很好。</S>
    >  tgt:     It's a beautiful day today .     </S>
    >  pred:    <S> It's a   beautiful day today .
    >    "<S>" predict "It's" ,   "<S> It's" predict "a" , "." predict "</S>"
    > 

  - $2*2$个list，$lang2idx，idx2lang$

  - $encoder$

    > input :  (N, T) 
    >
    > output : (N, T, num_units)   # num_units:隐层大小

    - embedding

      > input :  (N, T) 
      >
      > output : (N, T, num_units)\

      建立 $lookup\_table$ (vocab_size, num_units)

      zero_pad:第一行置0

      tf.nn.embedding_lookup(lookup_table, inputs)

      这里推测输出 inputs中idx对应的vector组成的tensor(N, T, num_units)

      缩放：scale  $outputs * (\sqrt{ num\_units })$

    - positional_encoding

      sin cos 常量 或者 调用embedding

    - dropout

      ​	0，其他缩放1/(1-rate)

      ​	maybe 可训练更新

    - 6层attention encoder self-attention

      > input :  (N, T, num_units)   
      >
      > output : (N, T, num_units)

      ```python
      multihead_attention(queries=self.enc, keys=self.enc, num_units=hp.hidden_units, num_heads=hp.num_heads, dropout_rate=hp.dropout_rate, is_training=is_training, causality=False)
          Q K V = dense num_units relu # (N, T_q, C) (N, T_k, C) (N, T_k, C)
          Q_ K_ V_ =	最后一维拆head分，在第一维拼接cat # (h*N, T_q, C/h) (h*N, T_k, C/h)  
          Q_ matmul V_  # (h*N, T_q, T_k)  矩阵乘法 Q 乘 k 转置
          scale：outputs / (num_units)**0.5
          Key Masking
           生成矩阵，key中最后一维加0的变为无穷小 -2**32+1 # (h*N, T_q, T_k)
          # causality参数告知我们是否屏蔽未来序列的信息
              # 首先定义一个和outputs后两维的shape相同shape（T_q,T_k）的一个张量（矩阵）。 
              # 然后将该矩阵转为三角阵tril。三角阵中，对于每一个T_q,
              # 凡是那些大于它角标的T_k值全都为
              # 0，这样作为mask就可以让query只取它之前的key（self attention中query即key）。
              # 由于该规律适用于所有query，接下来仍用tile扩展堆叠其第一个维度，构成masks，
              # shape为(h*N, T_q,T_k).
          softmax
          Query Masking # (h*N, T_q, T_k)
          dropout
          outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
          concat # (N, T_q, C)
          +querys # residual
          normalize # 随训练会变
      feedforward()
      	两层卷积之间加了relu非线性操作。
          之后是residual操作加上inputs残差
          然后是normalize
      
      ```

  - $decoder$

    - word embedding 

    - positional embedding 

    - dropout  # (N, T_q, C)

    - 6层attention decoder self-attention  encoder-decoder attention

      ```python
      self:
      	causality参数为True，以屏蔽未来的信息
      e-d:
      	query:decoder
      	keys: encoder
      	# causality设置为False 解码器中的信息都可以被用到
      feedforward:
      ```

      

  - 准确率 acc

  - smooth

    ​	self.y转为one_hot之后用module中定义的label_smoothing函数进行平滑操作 

    ​	((1-epsilon) * inputs) + (epsilon / K) ：0->很小的树 1->接近1的数

  - loss：

    ​	 预测值 和 平滑后的lable 交叉熵

    