---
title: Train in Deep Learning
date: 2019-04-18 14:43:57
header-img: /images/home-bg-o.jpg
tags:
    - 深度学习

---

> 自己看的记录文档

### epoch、steps

```python
total = total * epoch
train_steps = total*5/(batchsize*4)/workergpu
```

### finetuning

- finetuning是使用一样的词表进行训练
- 模型(output dir)里面放置预训练好的模型
- finetuning的train_steps是在原模型的train_steps上进行增加

###loss in T2T and Fairseq

T2T中的Loss是考虑之前所有epoch(batch?)的训练，然后经过平滑得出来的Loss

Fairseq则不同

> Todo: 计算细节

**经验：**

- loss(ppl)不再下降：训练差不多了
- Fairseq中loss可能最终掉到  valid_loss 4.08244 | valid_nll_loss 2.36351 这个数值
- T2T中的loss可能最终是不到2分

### train.sh

一个写的不错的训练脚本

```shell
#! /usr/bin/bash
set -e

######## hardware ########
# devices
dev=0,1,2,3,4,5,6,7
# how many percentages per gpu
gpu_fraction=0.95

######## dataset ########
# language: zh2en or en2zh
#lang=de2en
lang=en2de
# datatype= version + segmentation
datatype=v1-bpe40k
# dataset: cwmt/wmt
dataset=wmt14

######## parameters ########
# which model
model=transformer
#model=transformer_dla

# which hparams
param=transformer_base
#param=transformer_base_v2
#param=transformer_base_v3
#param=transformer_before
#param=transformer_before_shared25
#param=transformer_big_multistep2
#param=transformer_dla_base
#param=transformer_dla_base25_shared
#param=transformer_dla_base30_shared
#param=transformer_dla_base20_shared_filter4096
#param=transformer_dla_rpr_base25

######## required ########
# tag is the name of your experiments, must exist
#tag=sampling_r2l_v3_20m
#tag=sampling_r2l_dla20_v3_20m
#tag=base_r2l_dla_rpr25_20m
#tag=base_dla30_seed30_20m
#tag=base_before_shared25_seed2_20m
#tag=base_dla25_20m

#tag=finetune_distillation_v3_10m
#tag=finetune_distillation_dla30_10m
#tag=finetune_distillation_dla25_10m
#tag=finetune_distillation_dla_rpr25_2_10m
#tag=finetune_distillation_dla20_v3_2_10m
tag=base_textRepair
random_seed=2

#if [ $lang == "de2en" ]; then
#	# for CWMT zh-en task, training data contains CWMT + 30%WMT, about 1000w, epoch=15, need 130k steps
#        # 39207
#	if [ $dataset == "wmt" ]; then
#		train_step=100000
#	else
#		echo "unknow dataset:$dataset"
#		exit
#	fi
#elif [ $lang == "en2de" ]; then
#        # for CWMT or WMT en-zh task, training data contains CWMT + 30%WMT + xinhua, about 1693w, epoch=15, need 271k steps
#        if [ $dataset == "wmt" ]; then
#                train_step=100000
#        else
#                echo "unknow dataset:$dataset"
#                exit
#        fi
#fi

# dynamic hparams, e.g. change the batch size without the register in code, other_hparams='batch_size=2048'
#other_hparams='batch_size=2048'
other_hparams=

train_step=50000
# automatically set worker_gpu according to $dev
worker_gpu=`echo "$dev" | awk '{split($0,arr,",");print length(arr)}'`
# dir of training data
data_dir=./data/$lang/$datatype-$dataset/data-bin
# dir of models
output_dir=./output/$lang/$datatype-$dataset/$tag
if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi
# save train.sh
cp `pwd`/train_fairseq.sh $output_dir

cmd="python3 ../tensor2tensor/bin/t2t-trainer
--problem=translate_ende_wmt32k 
--model=$model 
--hparams_set=$param 
--data_dir=$data_dir 
--output_dir=$output_dir 
--worker_gpu_memory_fraction=$gpu_fraction
--train_steps=$train_step
--random_seed=$random_seed 
--worker_gpu=$worker_gpu
--local_eval_frequency=0"
if [ -n "$other_hparams" ]; then
	cmd=${cmd}" --hparams="${other_hparams}
fi

#echo "run command:"$cmd
# start training
#CUDA_VISIBLE_DEVICES=$dev PYTHONPATH=`pwd`/.. nohup $cmd exec 1> $output_dir/train.log exec 2>&1 &

# start multi-gpu evaluation on CPU
cmd="python3 -u fairseq/train.py 
$data_dir 
-a transformer --optimizer adam 
--lr 0.0007 
-s en -t de --label-smoothing 0.1 
--dropout 0.1 --max-tokens 4096 
--min-lr 1e-09 --lr-scheduler inverse_sqrt 
--weight-decay 0.0001 
--criterion label_smoothed_cross_entropy 
--max-update 100000 
--update-freq 1 --warmup-updates 4000 
--warmup-init-lr 1e-07 
--save-dir $output_dir
"
adam_betas="'(0.9,0.98)'"
cmd=${cmd}" --adam-betas "${adam_betas}
echo "run command:"$cmd
#CUDA_VISIBLE_DEVICES=$dev PYTHONPATH=`pwd`/.. nohup $cmd  $output_dir/train.log 2>&1 &
cmd="CUDA_VISIBLE_DEVICES=$dev PYTHONPATH=`pwd`/. nohup "${cmd}" > $output_dir/train.log 2>&1 &"
eval $cmd
echo $cmd
tail -f $output_dir/train.log

cmd="python3 ../tensor2tensor/bin/t2t-eval 
  --data_dir=$data_dir
  --problems=wmt_zhen_tokens_32k 
  --model=$model 
  --hparams_set=$param 
  --local_eval_frequency=0 
  --densenet=False 
  --eval_step=8 
  --locally_shard_to_cpu=False 
  --output_dir=$output_dir
  --train_steps=$train_step"
#CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=`pwd`/.. nohup $cmd exec 1> $output_dir/eval.log exec 2>&1 & 

# monitor training log
tail -f $output_dir/train.log

```

