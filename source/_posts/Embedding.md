---
layout: word
title: Embedding
date: 2019-09-04 16:16:16
tags:
---

关于Embeddings的摘抄笔记

<!--more-->
参考：https://github.com/imhuay/Algorithm_Interview_Notes-Chinese/blob/master/B-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/B-%E4%B8%93%E9%A2%98-%E8%AF%8D%E5%90%91%E9%87%8F.md


#### 1 理解
##### 定义
- 是神经语言模型的副产品。
- 可以有词向量，字向量，句向量，文档向量。

##### 分布式假设

1 Distributional Representation是从分布式假设（即如果两个词的上下文相似，那么这两个词也是相似的）的角度，指的是一类获取文本表示的方法，

一些分布式表示方法
- 潜在语义分析模型 Latent Semantic Analysis， LSA
    - svd分解
- 潜在狄利克雷分配模型 Latent Dirchler Allocation LDA, 主题模型
- 神经网络，深度学习

2 Distributed Representation是文本的一种表示形式，具体为稠密、低维、连续的向量。向量的每一维都表示文本的某种潜在的语法或语义特征。Distributed Representation翻译为分散式表示可能理解起来会更明确些。

#### 2 Word2Vec
- 是一个神经网络模型，为了得到更快更好的词向量
- Word2Vec 提供了两套模型：CBOW 和 Skip-Gram(SG)
    - CBOW 在已知 context(w) 的情况下，预测 w
    - SG 在已知 w 的情况下预测 context(w)
- 除了两套模型，Word2Vec 还提供了两套优化方案，分别基于 Hierarchical Softmax (层次SoftMax) 和 Negative Sampling (负采样)

###### 基于层次 SoftMax 的 CBOW 模型
<div style="width: 800px; margin: auto">![avater](1.PNG)</div>

###### 层次 SoftMax 的正向传播
- 输入时context（“足球”）
- 以词表中词作为叶子节点，各词的出现频率作为权重!像图里叶子节点15是我出现的频率or次数
<div style="width: 800px; margin: auto">![avater](2.PNG)</div>

###### 层次 SoftMax 的反向传播
- https://blog.csdn.net/itplus/article/details/37969979

###### 基于层次 Softmax 的 Skip-gram 模型
<div style="width: 800px; margin: auto">![avater](3.PNG)</div>

###### 基于负采样的 CBOW 和 Skip-gram

- 层次 Softmax 还不够简单，于是提出了基于负采样的方法进一步提升性能
- 负采样（Negative Sampling）是 NCE(Noise Contrastive Estimation) 的简化版本，NCE:https://blog.csdn.net/littlely_ll/article/details/79252064
- CBOW 的训练样本是一个 (context(w), w) 二元对；对于给定的 context(w)，w 就是它的正样本，而其他所有词都是负样本。
- 如果不使用负采样，即 N-gram 神经语言模型中的做法，就是对整个词表 Softmax 和交叉熵
- 负采样相当于**选取所有负例中的一部分**作为负样本，从而减少计算量
- Skip-gram 模型同理

###### 负采样算法
- 负采样算法，即对给定的 单词w ，生成相应负样本的方法
- 最简单的方法是随机采样，但这会产生一点问题，词表中的词出现频率并不相同
    - 如果不是从词表中采样，而是从语料中采样；显然，那些高频词被选为负样本的概率要大于低频词
- 因此，负采样算法实际上就是一个带权采样过程

<div style="width: 800px; margin: auto">![avater](4.PNG)</div>

###### 一些源码细节
<div style="width: 800px; margin: auto">![avater](5.PNG)</div>

<div style="width: 800px; margin: auto">![avater](6.PNG)</div>


#### 3 Glove


##### 共现矩阵


##### 基本思想

##### 目标函数

##### 与Word2vec的区别