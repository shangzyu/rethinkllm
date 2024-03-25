# Easyllm
>记录一些LLM笔记

## 大模型理论
### 几种attention
multi head attention、multi query attention、group query attention：  
[为什么现在大家都在用 MQA 和 GQA？](https://lonepatient.top/2023/08/03/MHA_MQA_GQA.html)  
[线性Attention的探索：Attention必须有个Softmax吗？](https://spaces.ac.cn/archives/7546)
### 位置编码
ROPE

### 激活函数
GELU

### 参数量
[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)  
[LLM大模型之精度问题（FP16，FP32，BF16）详解与实践](https://zhuanlan.zhihu.com/p/657886517)  
[LLM大模型之不同精度下显存占用与相互转换实践](https://zhuanlan.zhihu.com/p/658343628) fp16/fp32/bf16之间的相互转换，尾数位直接去掉，指数位-127+15
### Normalization

### tokenizer相关

## 大模型结构
### 组成部分
[为什么现在的LLM都是Decoder-only的架构？](https://spaces.ac.cn/archives/9529/comment-page-1)
### 常见模型
LLama  
Mixtral  
GLM



## 模型训练


## 模型量化

## 模型部署

## Prompt Engineering

## RAG