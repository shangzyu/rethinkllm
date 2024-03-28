# Easyllm
>记录一些LLM好文

## 大模型综述
[通向AGI之路：大型语言模型（LLM）技术精要](https://zhuanlan.zhihu.com/p/597586623) 最好的综述，解释了LLM中attention和FFN参数蕴含知识的区别  
[人工智能对齐是什么、以及我为什么选择研究它？](https://zhuanlan.zhihu.com/p/655464730)



## LLM 组成原理
### Tokenizer 相关
[大模型基础知识系列：从头训练一个自己的Tokenizer](https://zhuanlan.zhihu.com/p/625715830)
[没有思考过 Embedding，不足以谈 AI](https://zhuanlan.zhihu.com/p/643560252)

### Attention 相关
Multi Head Attention(MHA)->Multi Query Attention(MQA)->Group Query Attention(GQA)  
[为什么现在大家都在用 MQA 和 GQA？](https://lonepatient.top/2023/08/03/MHA_MQA_GQA.html)  
[线性Attention的探索：Attention必须有个Softmax吗？](https://spaces.ac.cn/archives/7546)  
[为什么现在的LLM都是Decoder-only的架构？](https://spaces.ac.cn/archives/9529/comment-page-1) 区分了encoder和decoder，encoder只是具有双向注意力的decoder

### 位置编码
[让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130)  
绝对位置编码  
Sinusoidal  
相对位置编码  
RoPE  
[Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265)  
[十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)  讲清楚了位置编码的公式，如何从二维扩展到多维  
[Transformer升级之路：12、无限外推的ReRoPE？](https://spaces.ac.cn/archives/9708)
[再论大模型位置编码及其外推性（万字长文）](https://zhuanlan.zhihu.com/p/675243992)  
### 激活函数
GELU/Swish/SwiGLU  
[大模型基础｜激活函数｜从ReLU 到SwiGLU](https://zhuanlan.zhihu.com/p/650237644)

### 参数量/计算量
[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)  
[LLM大模型之精度问题（FP16，FP32，BF16）详解与实践](https://zhuanlan.zhihu.com/p/657886517)  
[LLM大模型之不同精度下显存占用与相互转换实践](https://zhuanlan.zhihu.com/p/658343628) fp16/fp32/bf16之间的相互转换，尾数位直接去掉，指数位-127+15  
[浅谈后向传递的计算量大约是前向传递的两倍](https://zhuanlan.zhihu.com/p/675517271)  
[LLM（二十）：漫谈 KV Cache 优化方法，深度理解 StreamingLLM](https://zhuanlan.zhihu.com/p/659770503)
[为什么大模型输入输出往往只有2K, 4K token?](https://www.zhihu.com/question/606514058)  
### Normalization  
Post-LN&Pre-LN  
Layer Norm&RMS Norm  
[[论文笔记]RMSNorm：Root Mean Square Layer Normalization](https://zhuanlan.zhihu.com/p/669071548) RMS Norm就是均值为0的Layer Norm







## LLM 结构
### 组成部分
[为什么现在的LLM都是Decoder-only的架构？](https://spaces.ac.cn/archives/9529/comment-page-1) 区分了encoder和decoder，encoder只是具有双向注意力的decoder
### 常见模型
Qwen1.5  
Mixtral  
[Mixtral-8x7B 模型挖坑](https://zhuanlan.zhihu.com/p/674751021)  写的很好的大模型逆向工程，解释了每个组件的作用   
[混合专家模型 (MoE) 详解](https://huggingface.co/blog/zh/moe)  
[Mixtral 8✖️7B=56B？错！一文带你看清Mixtral内部结构及参数计算](https://zhuanlan.zhihu.com/p/673527090)  
[群魔乱舞：MoE大模型详解](https://www.zhihu.com/tardis/zm/art/677638939?source_id=1003)  
[从开源LLM中学模型架构优化-Mistral 7B](https://zhuanlan.zhihu.com/p/658911982)  
LLama  
[Llama 2 详解](https://mp.weixin.qq.com/s?__biz=MzU2NzE2MjE2Nw==&mid=2247484226&idx=1&sn=b5b26468548f4dbb3e6d2bd52b2b7feb&chksm=fca0271acbd7ae0cd591a0314d00ece2b696017ad26709b14313b942ce4077cffc01612fcb10&scene=21#wechat_redirect)   
[大模型升级与设计之道：ChatGLM、LLAMA、Baichuan及LLM结构解析](https://zhuanlan.zhihu.com/p/651747035)  
GLM  
[预训练语言模型：GLM](https://zhuanlan.zhihu.com/p/641499380)
### 限制
[从0开始实现LLM：4、长上下文优化（理论篇）](https://zhuanlan.zhihu.com/p/683731440)





## 模型训练
### Pretrain
### SFT
### RLHF  
[强化学习小记——观其大略](https://zhuanlan.zhihu.com/p/646787054)  
[ChatGPT 背后的“功臣”——RLHF 技术详解
](https://mp.weixin.qq.com/s/TLQ3TdrB5gLb697AFmjEYQ)
PPO  
[【RLHF】怎样让 PPO 训练更稳定？早期人类征服 RLHF 的驯化经验](https://zhuanlan.zhihu.com/p/666455333)  
DPO  
[DPO——RLHF 的替代之《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》论文阅读](https://zhuanlan.zhihu.com/p/634705904)
### PEFT
Lora系列  
[大模型参数高效微调技术原理综述（五）-LoRA、AdaLoRA、QLoRA](https://zhuanlan.zhihu.com/p/636215898)  
[QLoRA（Quantized LoRA）详解](https://zhuanlan.zhihu.com/p/666234324)  
[配置不同的学习率，LoRA还能再涨一点？](https://spaces.ac.cn/archives/10001)  
P-Tuning  
Prefix Tuning
### 分布式训练
数据并行  
[deepspeed入门教程](https://zhuanlan.zhihu.com/p/630734624)  
[DeepSpeed之ZeRO系列：将显存优化进行到底](https://zhuanlan.zhihu.com/p/513571706)  
[DeepSpeed配置文件Json参数解析](https://zhuanlan.zhihu.com/p/645627795)  
[数据并行Deep-dive: 从DP 到 Fully Sharded Data Parallel （FSDP）完全分片数据并行](https://zhuanlan.zhihu.com/p/485208899)  
[如何评价微软开源的分布式训练框架deepspeed？](https://www.zhihu.com/question/371094177/answer/3330130413)
[[LLM]大模型训练(二)--DeepSpeed使用](https://blog.csdn.net/zwqjoy/article/details/135314202)  
[图解大模型训练之：数据并行下篇( DeepSpeed ZeRO，零冗余优化)](https://zhuanlan.zhihu.com/p/618865052)
[DeepSpeed配置文件Json参数解析](https://zhuanlan.zhihu.com/p/645627795)
张量并行   
[图解大模型训练之：张量模型并行(TP)，Megatron-LM](https://zhuanlan.zhihu.com/p/622212228)
### Trainer
[LLM大模型之Trainer以及训练参数](https://zhuanlan.zhihu.com/p/662619853)
### 评估指标
[一文带你理解｜NLP评价指标 BLEU 和 ROUGE（无公式）](https://zhuanlan.zhihu.com/p/647310970)





## Prompt Engineering
[构建高性能Prompt之路——结构化Prompt ](https://lonepatient.top/2023/08/01/Structured_prompts.html)
### RAG
### COT
[大模型思维链（Chain-of-Thought）技术原理](https://zhuanlan.zhihu.com/p/629087587)





## 模型推理
[大模型推理框架概述](https://juejin.cn/post/7286676030965317668)  

生成和采样  
[LLM大语言模型之Generate/Inference（生成/推理）中参数与解码策略原理及其代码实现](https://zhuanlan.zhihu.com/p/653926703)  
[LLM（大语言模型）解码时是怎么生成文本的？](https://www.likecs.com/show-308663700.html)  
[【NLP学习】自然语言生成中的top-k, top-p, typical采样方法的实现](https://zhuanlan.zhihu.com/p/560847355)  
[如何通俗的理解beam search？](https://zhuanlan.zhihu.com/p/82829880)    
vLLM  
[LLM推理2：vLLM源码学习](https://zhuanlan.zhihu.com/p/643336063)  
[LLM推理4：vllm和HF推理结果不一致](https://zhuanlan.zhihu.com/p/658780653)  
[LLM（十七）：从 FlashAttention 到 PagedAttention, 如何进一步优化 Attention 性能](https://zhuanlan.zhihu.com/p/638468472)





## 模型量化
[大模型量化概述](https://blog.csdn.net/scgaliguodong123_/article/details/136176355)  
AWQ  
[AWQ：用于 LLM 压缩和加速的激活感知权重量化](https://zhuanlan.zhihu.com/p/669061765)
GPTQ




## 数据工程
数据预处理  
[HuggingFace | 在HuggingFace中预处理数据的几种方式](https://zhuanlan.zhihu.com/p/341994096)  
[大部分的大模型(LLM)采用左填充(left-padding)的原因](https://zhuanlan.zhihu.com/p/646852375)  
数据工程  
[研发大模型的血液--万字长文详谈数据工程](https://mp.weixin.qq.com/s/izePeavfxezfEkkPzgMmjQ)