# Easyllm
>记录一些LLM好文

## 大模型理论
### 几种attention
multi head attention、multi query attention、group query attention：  
[为什么现在大家都在用 MQA 和 GQA？](https://lonepatient.top/2023/08/03/MHA_MQA_GQA.html)  
[线性Attention的探索：Attention必须有个Softmax吗？](https://spaces.ac.cn/archives/7546)
### 位置编码
ROPE  
[十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)  
[Transformer升级之路：12、无限外推的ReRoPE？](https://spaces.ac.cn/archives/9708)
### 激活函数
GELU/Swish  
[大模型基础｜激活函数｜从ReLU 到SwiGLU](https://zhuanlan.zhihu.com/p/650237644)

### 参数量
[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)  
[LLM大模型之精度问题（FP16，FP32，BF16）详解与实践](https://zhuanlan.zhihu.com/p/657886517)  
[LLM大模型之不同精度下显存占用与相互转换实践](https://zhuanlan.zhihu.com/p/658343628) fp16/fp32/bf16之间的相互转换，尾数位直接去掉，指数位-127+15
### Normalization
Post-LN&Pre-LN  
Layer Norm  
RMS Norm  
[Llama 美洲鸵（大羊驼）改进之一：均方层归一化RMSNorm](https://blog.csdn.net/qq_39970492/article/details/131125752)

### Tokenizer相关
[大模型基础知识系列：从头训练一个自己的Tokenizer](https://zhuanlan.zhihu.com/p/625715830)

## 大模型结构
### 组成部分
[为什么现在的LLM都是Decoder-only的架构？](https://spaces.ac.cn/archives/9529/comment-page-1)
### 常见模型
Mixtral  
[Mixtral-8x7B 模型挖坑](https://zhuanlan.zhihu.com/p/674751021)  写的很好的大模型逆向工程，解释了每个组件的作用  
LLama  
[Llama 2 详解](https://mp.weixin.qq.com/s?__biz=MzU2NzE2MjE2Nw==&mid=2247484226&idx=1&sn=b5b26468548f4dbb3e6d2bd52b2b7feb&chksm=fca0271acbd7ae0cd591a0314d00ece2b696017ad26709b14313b942ce4077cffc01612fcb10&scene=21#wechat_redirect)   
GLM  
[预训练语言模型：GLM](https://zhuanlan.zhihu.com/p/641499380)
### 限制
[从0开始实现LLM：4、长上下文优化（理论篇）](https://zhuanlan.zhihu.com/p/683731440)


## 模型训练
Deepspeed  
[deepspeed入门教程](https://zhuanlan.zhihu.com/p/630734624)  
[DeepSpeed之ZeRO系列：将显存优化进行到底](https://zhuanlan.zhihu.com/p/513571706)  
[DeepSpeed配置文件Json参数解析](https://zhuanlan.zhihu.com/p/645627795)


## 模型推理

[大模型推理框架概述](https://juejin.cn/post/7286676030965317668)  
vLLM  
[LLM推理2：vLLM源码学习](https://zhuanlan.zhihu.com/p/643336063)  
[LLM推理4：vllm和HF推理结果不一致](https://zhuanlan.zhihu.com/p/658780653)

## 模型量化

## Prompt Engineering

## RAG