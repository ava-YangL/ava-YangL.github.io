---
title: POS tagging
date: 2019-07-16 14:41:06
tags:
---

#### Question
1 CRFSuite内部对于字符型的特征函数是怎么处理的
2 Lbfgs优化算法

<!--more-->



Index
---
<!-- TOC -->
- [POS Tagging可用模型](#POSTagging可用模型)
    - [1 HMM](#1HMM)
    - [2 CRF](#2CRF)
    - [3 LBFGS](#3LBFGS)
    - [4 ELMO BERT XLNET](#4ELMOBERTXLNET)
- [NLP发展趋势](#NLP发展趋势)
    - [1 2018黄金十年](#12018黄金十年)
    - [2 2019对话MSRA副院长周明](#22019对话MSRA副院长周明)


- [NLP 概述](#NLP概述)
    - [解决 NLP 问题的一般思路](#解决nlp问题的一般思路)
    - [NLP 的历史进程](#nlp的历史进程)
    - [Seq2Seq 模型](#seq2seq模型)
    - [评价机制](#评价机制)
        - [困惑度 (Perplexity, PPX)](#困惑度perplexityppx)
        - [BLEU](#bleu)
        - [ROUGE](#rouge)
- [语言模型](#语言模型)
    - [XX 模型的含义](#xx模型的含义)
    - [概率/统计语言模型 (PLM, SLM)](#概率统计语言模型plmslm)
        - [参数的规模](#参数的规模)
        - [可用的概率模型](#可用的概率模型)
    - [N-gram 语言模型](#ngram语言模型)
        - [可靠性与可区别性](#可靠性与可区别性)
        - [OOV 问题](#oov-问题)
        - [平滑处理 TODO](#平滑处理-todo)
    - [神经概率语言模型 (NPLM)](#神经概率语言模型-nplm)
        - [N-gram 神经语言模型](#n-gram-神经语言模型)
            - [N-gram 神经语言模型的网络结构](#n-gram-神经语言模型的网络结构)
        - [模型参数的规模与运算量](#模型参数的规模与运算量)
        - [相比 N-gram 模型，NPLM 的优势](#相比-n-gram-模型nplm-的优势)
        - [NPLM 中的 OOV 问题](#nplm-中的-oov-问题)
- [句嵌入](#句嵌入)
    - [基线模型](#基线模型)
    - [词袋模型](#词袋模型)
    - [无监督模型](#无监督模型)
    - [有监督模型](#有监督模型)
    - [多任务学习](#多任务学习)



<!-- /TOC -->

# POSTagging可用模型

#### 1HMM

隐马尔科夫模型(HMM)的三大基本问题与解决方案：
- 对于一个观察序列匹配最可能的系统——评估，使用前向算法（forward algorithm）；
- 对于已生成的一个观察序列，确定最可能的隐藏状态序列——解码，使用维特比算法（Viterbi algorithm）；
- 对于已生成的观察序列，决定最可能的模型参数——学习，使用Baum-Welch算法。

<!--more-->

方法：
1 获得训练好的模型参数：隐藏状态+观测符号+状态转移概率+状态随观测符号的发射概率
- 标注了的话很好算
- 没标注的话，通过一些辅助资源，利用前向-后向算法学习一个hmm

2 维特比算法
- 根据模型和观测序列，确定隐藏状态

可以参见： https://github.com/IdearHui/posTag
<div style="width: 600px; margin: auto">![avater](1.jpg)</div>




#### 2CRF

可以参见：https://www.cnblogs.com/pinard/p/7048333.html

- 随机场： 若干个位置组成的整体，当给每一个位置中按照某种分布随机赋予一个值之后，其全体就叫做随机场。还是举词性标注的例子：假如我们有一个十个词形成的句子需要做词性标注。这十个词每个词的词性可以在我们已知的词性集合（名词，动词...)中去选择。当我们为每个词选择完词性后，这就形成了一个随机场。
- 马尔科夫随机场： 是随机场的特例，它假设随机场中某一个位置的赋值仅仅与和它相邻的位置的赋值有关，和与其不相邻的位置的赋值无关。继续举十个词的句子词性标注的例子：　如果我们假设所有词的词性只和它相邻的词的词性有关时，这个随机场就特化成一个马尔科夫随机场。比如第三个词的词性除了与自己本身的位置有关外，只与第二个词和第四个词的词性有关。　
- 条件随机场CRF:假设马尔科夫随机场中只有X和Y两种变量，X一般是给定的，而Y一般是在给定X的条件下我们的输出。这样马尔科夫随机场就特化成了条件随机场。在我们十个词的句子词性标注的例子中，**X是词，Y是词性**。因此，如果我们假设它是一个马尔科夫随机场，那么它也就是一个CRF。
对于CRF，我们给出准确的数学语言描述：
设X与Y是随机变量，P(Y|X)是给定X时Y的条件概率分布，若随机变量Y构成的是一个马尔科夫随机场，则称条件概率分布P(Y|X)是条件随机场。
例如特征模板可以是这样的
- 线性链条件随机场 ：X和Y有相同的结构的CRF就构成了线性链条件随机场
我们再来看看 linear-CRF的数学定义：
　　设X=(X1,X2,...Xn),Y=(Y1,Y2,...Yn)均为线性链表示的随机变量序列，在给定随机变量序列X的情况下，随机变量Y的条件概率分布P(Y|X)构成条件随机场，即满足马尔科夫性：
P(Yi|X,Y1,Y2,...Yn)=P(Yi|X,Yi−1,Yi+1)
　　则称P(Y|X)为线性链条件随机场
- 线性链条件随机场参数化： 就是转换为机器学习模型，通过特征函数和权重系数定义：当前节点特征函数+上下文节点特征函数（没有不相邻节点之间的特征函数，是因为我们的linear-CRF满足马尔科夫性。）
- 特征函数的取值只能是0/1

<div style="width: 600px; margin: auto">![avater](1.png)</div>

去看上面的博客吧

```c
# https://nlpforhackers.io/crf-pos-tagger/
  return {
        'word': sentence[index],
        'is_first': index == 0,     #是第一个单词吗
        'is_last': index == len(sentence) - 1,   #是最后一个单词吗
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],  #是否首字母大写
        'is_all_caps': sentence[index].upper() == sentence[index],    #是否大写后还是他本身
        'is_all_lower': sentence[index].lower() == sentence[index],    #是否小写后还是他本身
        'prefix-1': sentence[index][0],     #比如Influential  I
        'prefix-2': sentence[index][:2],   #In
        'prefix-3': sentence[index][:3],  #Inf
        'suffix-1': sentence[index][-1],  #l
        'suffix-2': sentence[index][-2:],  #al
        'suffix-3': sentence[index][-3:],  #ial
        'prev_word': '' if index == 0 else sentence[index - 1],  #上一个单词
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],  #下一个单词
        'has_hyphen': '-' in sentence[index],   # 有没有-这个符号
        'is_numeric': sentence[index].isdigit(),  # 是不是数字
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:] # 除了第一个字母外 有没有大写的字母
    }
    ```

LBFGS的优化方法，这个是啥-.-
https://www.cnblogs.com/alexanderkun/p/4024600.html

#### 3LBFGS

##### 3.1 梯度向量和海森矩阵
设自变量：x = ( x1, x2 ,⋯, xn )T
*一维因变量*：一维f(x)：
一阶导数构成的向量为梯度向量 g(x) （nx1）
二阶导数构成的矩阵为Hessian矩阵(海森矩阵)(nxn)
- https://www.jianshu.com/p/7bf1d37751b3


#### 4ELMOBERTXLNET
WordEmbedding， 将符号化的中文或者英文或者拉丁文，嵌入到数学空间的表示，这种就叫做词嵌入（Word Embedding）
- Word2Vec，词嵌入的一种。
Glove？
gensim？
他们两个不能处理好有上下文的语义关系。
https://zhuanlan.zhihu.com/p/26306795

##### 4.1 Embeddings from Language Models, ELMO

- 18年6月发表在NAACL,是一个用来获得词向量的语言模型。
- 能根据上下文处理一些一词多义的情况
- 可以用大语料库训练他，再迁移到别的上面去用。
- 和Word2Vec相比，ELMO是在整个语料库去学习的，而Word2Vec是中心词的上下文窗口去学习的。

<div style="width: 800px; margin: auto">![avater](elmo.PNG)</div>


##### 4.2 GPT (1.0 2.0 3.0?)

<div style="width: 800px; margin: auto">![avater](bert.PNG)</div>

<div style="width: 800px; margin: auto">![avater](bert2.PNG)</div>



以下内容来自：

https://github.com/imhuay/Algorithm_Interview_Notes-Chinese
------------------------------------

# NLP发展趋势
===

## 12018黄金十年

#### 1 新的发展基础
- 各个行业的**文本大数据**：采集、加工、入库；
- **需求**：来自搜索引擎、客服、商业智能、语音助手、翻译、教育、法律、金融等领域对NLP的需求会大幅度上升，更高的要求；
- 文本数据和**语音**、**图像数据**的**多模态融合**成为未来机器人的刚需。

#### 2 研究热点
1. 将**知识**和**常识**引入目前基于数据的学习系统中；
1. **低资源**的NLP任务的学习方法；
1. **上下文**建模、**多轮**语义理解；
1. 基于语义分析、知识和常识的**可解释 NLP**。
>- NLP场景应用中的可解释性，可以直观的表述为，业务人员可以看懂的、符合业务知识预期的NLP处理结果。
>-  可解释性是存在于场景应用和NLP技术之间的最大的鸿沟。包括当前流行的深度学习算法，在场景应用下遇到的最大的尴尬也是可解释性差，即使业务人员看到一个明显的错误，也很难调整模型以更正。经常能听到这样的对话：

#### 3 重要进展
- 神经机器翻译?
    - （Hassan et al., 2018）高效利用大规模单语数据的联合训练、对偶学习技术；解决曝光偏差问题的一致性正则化技术、推敲网络
- 智能人机交互  seq2seq
    - 聊天系统架构
    > - 阅读（Encoder）:用户上下文，小冰上下文，用户当前输入。
    > - 阅读那一层输入各自的lstm+对话情感+用户画像
    > - 提炼（Attention）
    > - Decoder（回答）

    - 三层引擎：
        - 第一层，通用聊天机器人；
        - 第二层，搜索和问答（Infobot）；
        - 第三层，面向特定任务对话系统（Bot）
- 机器阅读理解
    - NL-Net
    - **BERT**
- 机器创作
    - 对联、
    - **歌词**的机器创作过程：
        - 确定**主题**，比如希望创作一首与“秋”、“岁月”、“沧桑”、“感叹”相关的歌；
        - 利用**词向量表示技术**找到相关的词，有“秋风”、“流年”、“岁月”、“变迁”等；
        - 通过**序列到序列**的神经网络，用歌词的上一句去生成下一句；如果是第一句，则用一个特殊的序列作为输入去生成第一句歌词
    - **谱曲**：类似一个**翻译**过程，但更为严格

#### 4 值得关注的 NLP 技术
- **预训练神经网络**
    - **ElMo**、**BERT** 等模型
    - 在什么**粒度**（word，sub-word，character）上进行预训练，用什么结构的语言模型（LSTM，**Transformer**等）训练，在什么样的数据上（不同体裁的文本）进行训练，以及如何将预训练的模型应用到具体任务，都是需要继续研究的问题。
- 低资源 NLP 任务
    - **无监督、半监督**学习
    - **迁移学习、多任务**学习
- 迁移学习
    - 不同的 NLP 任务虽然采用各自不同类型的数据进行模型训练，但在**编码器（Encoder）端往往是同构**的，即都会将输入的词或句子转化为对应的向量表示，然后再使用**各自的解码器**完成后续**翻译、改写和答案生成**等任务。
    - 因此，可以将通过不同任务训练得到的编码器看作是不同任务对应的一种向量表示模型；然后通过迁移学习（Transfer Learning）的方式将这类信息迁移到目前关注的目标任务上来
- 多任务学习
    - 多任务学习（Multi-task Learning）可通过端到端的方式，直接在主任务中引入其他辅助任务的监督信息，用于保证模型能够学到不同任务间共享的知识和信息；
    - McCann 等提出了利用**问答框架**使用多任务学习训练十项自然语言任务
        > [一个模型搞定十大自然语言任务](https://www.toutiao.com/a6569393480089469454)
- 知识和常识的引入
    - 应用：机器阅读理解、语义分析
    - 领域知识：**维基百科、知识图谱**
    - **常识**的引入（缺乏深入研究）
- **多模态**学习
    - 视觉问答
        - 基于问题生成的**视觉问答**方法（Li et al., 2018）
        - 基于场景图生成的视觉问答方法（Lu et al., 2018）
    - 视频问答


## 22019对话MSRA副院长周明
> https://www.toutiao.com/i6656937082805551624 - 头条

#### 1 MSRA 的重点
- 机器阅读理解（MRC）
- 神经机器翻译（NMT）
    - 联合训练、对偶学习 ？
    - 一致性规范、推敲网络 ？
- **语法检查**（Grammar Check）
    - 自动生成训练语料、多次解码逐轮求优 ？
- 文本语音转换（Text To Speech, TTS）
- 机器创作（写诗、谱曲、新闻）

#### 2  NLP 领域进展
- 新的神经 NLP 模型
- 以 BERT 为首的**预训练模型**
    - 大规模语料所蕴含的普遍语言知识与具体应用场景相结合的潜力
- **低资源** NLP 任务

#### 3 中文 NLP 的突破
- 中文阅读理解
- 以中文为中心的机器翻译
- 聊天系统
    - 多模态聊天（小冰）？
- 语音对话
    - 搜索引擎、语音助手、智能音箱、物联网、电子商务、智能家居等

#### 4  2019 研究热点
> [NLP将迎来黄金十年](#2019预见未来｜nlp将迎来黄金十年)一文中有更详细的介绍，包括最新的论文推荐

1. 预训练模型
    - 上下文建模
1. 基于语义分析、知识和常识的可解释 NLP
    - 将知识和常识引入到目前基于数据的学习模型中
1. 多模态融合
    - 文本数据和语音、图像数据的多模态融合是未来机器人的刚需
1. 低资源 NLP 任务（无语料或者小语料的场景）
    - 半监督、无监督学习方法
    - Transfer Learning、Multi-task Learning
    - **语言学**在 NLP 研究中的作用
        > 语言学家在自然语言处理研究中大有可为 - 冯志伟
1. 多模态融合
    - 在神经网络的框架下，可以用统一的模式来对多模态（语言、文字、图像、视频）进行建模（编码和解码），从而实现端到端的学习
    - 应用：Capturing、CQA/VQA、机器创作


NLP-NLP基础
===

Index
---
<!-- TOC -->

- [NLP 概述](#nlp概述)
    - [解决 NLP 问题的一般思路](#解决-nlp-问题的一般思路)
    - [NLP 的历史进程](#nlp-的历史进程)
    - [Seq2Seq 模型](#seq2seq-模型)
    - [评价机制](#评价机制)
        - [困惑度 (Perplexity, PPX)](#困惑度-perplexity-ppx)
        - [BLEU](#bleu)
        - [ROUGE](#rouge)
- [语言模型](#语言模型)
    - [XX 模型的含义](#xx-模型的含义)
    - [概率/统计语言模型 (PLM, SLM)](#概率统计语言模型-plm-slm)
        - [参数的规模](#参数的规模)
        - [可用的概率模型](#可用的概率模型)
    - [N-gram 语言模型](#n-gram-语言模型)
        - [可靠性与可区别性](#可靠性与可区别性)
        - [OOV 问题](#oov-问题)
        - [平滑处理 TODO](#平滑处理-todo)
    - [神经概率语言模型 (NPLM)](#神经概率语言模型-nplm)
        - [N-gram 神经语言模型](#n-gram-神经语言模型)
            - [N-gram 神经语言模型的网络结构](#n-gram-神经语言模型的网络结构)
        - [模型参数的规模与运算量](#模型参数的规模与运算量)
        - [相比 N-gram 模型，NPLM 的优势](#相比-n-gram-模型nplm-的优势)
        - [NPLM 中的 OOV 问题](#nplm-中的-oov-问题)

<!-- /TOC -->

# NLP概述

## 解决 NLP 问题的一般思路
```tex
这个问题人类可以做好么？
  - 可以 -> 记录自己的思路 -> 设计流程让机器完成你的思路
  - 很难 -> 尝试从计算机的角度来思考问题
```

## NLP 的历史进程
- **规则系统**
    - 正则表达式/自动机
    - 规则是固定的
    - **搜索引擎**
        ```tex
        “豆瓣酱用英语怎么说？”
        规则：“xx用英语怎么说？” => translate(XX, English)
        
        “我饿了”
        规则：“我饿（死）了” => recommend(饭店，地点)
        ```
- **概率系统**
    - 规则从数据中**抽取**
    - 规则是有**概率**的
    - 概率系统的一般**工作方式**
        ```tex
        流程设计
        收集训练数据
            预处理
            特征工程
                分类器（机器学习算法）
                预测
                    评价
        ```
        - 最重要的部分：数据收集、预处理、特征工程
    - 示例
        ```tex
        任务：
        “豆瓣酱用英语怎么说” => translate(豆瓣酱，Eng)

        流程设计（序列标注）：
        子任务1： 找出目标语言 “豆瓣酱用 **英语** 怎么说”
        子任务2： 找出翻译目标 “ **豆瓣酱** 用英语怎么说”

        收集训练数据：
        （子任务1）
        “豆瓣酱用英语怎么说”
        “茄子用英语怎么说”
        “黄瓜怎么翻译成英语”
        
        预处理：
        分词：“豆瓣酱 用 英语 怎么说”

        抽取特征：
        （前后各一个词）
        0 茄子：    < _ 用
        0 用：      豆瓣酱 _ 英语
        1 英语：    用 _ 怎么说
        0 怎么说：  英语 _ >

        分类器：
        SVM/CRF/HMM/RNN

        预测：
        0.1 茄子：    < _ 用
        0.1 用：      豆瓣酱 _ 英语
        0.7 英语：    用 _ 怎么说
        0.1 怎么说：  英语 _ >

        评价：
        准确率
        ```
- **概率系统**的优/缺点
    - `+` **规则更加贴近**于真实事件中的规则，因而效果往往比较好
    - `-` 特征是由专家/人指定的；
    - `-` 流程是由专家/人设计的；
    - `-` 存在独立的**子任务**

- **深度学习**
    - 深度学习相对概率模型的优势
        - 特征是由专家指定的 `->` 特征是由深度学习自己提取的
        - 流程是由专家设计的 `->` 模型结构是由专家设计的
        - 存在独立的子任务 `->` End-to-End Training

## Seq2Seq 模型
- 大部分自然语言问题都可以使用 Seq2Seq 模型解决
   <div style="width: 600px; margin: auto">![avater](seq1.png)</div>


## 评价机制

### 困惑度 (Perplexity, PPX)
> [Perplexity](https://en.wikipedia.org/wiki/Perplexity) - Wikipedia
- 在信息论中，perplexity 用于度量一个**概率分布**或**概率模型**预测样本的好坏程度

<h3>基本公式</h3>

- **概率分布**（离散）的困惑度
     <div style="width: 600px; margin: auto">![avater](eq1.PNG)</div>
    
    > 其中 `H(p)` 即**信息熵** (注意熵越大，不确定性越大，**2的熵次方**)

- **概率模型**的困惑度
    <div style="width: 600px; margin: auto">![avater](eq2.PNG)</div>
    > 通常 `b=2`
  
- **指数部分**也可以是**交叉熵**的形式，此时困惑度相当于交叉熵的指数形式
   <div style="width: 600px; margin: auto">![avater](eq3.PNG)</div>

    > 其中 `p~` 为**测试集**中的经验分布——`p~(x) = n/N`，其中 `n` 为 x 的出现次数，N 为测试集的大小

**语言模型中的 PPX**
- 在 **NLP** 中，困惑度常作为**语言模型**的评价指标
   <div style="width: 600px; margin: auto">![avater](eq4.PNG)</div>

- 直观来说，就是下一个**候选词数目**的期望值——

  如果不使用任何模型，那么下一个候选词的数量就是整个词表的数量；通过使用 `bi-gram`语言模型，可以将整个数量限制到 `200` 左右

### BLEU
> [一种机器翻译的评价准则——BLEU](https://blog.csdn.net/qq_21190081/article/details/53115580) - CSDN博客 
> https://zhuanlan.zhihu.com/p/39100621
<div style="width: 900px; margin: auto">![avater](eq5.PNG)</div>

### ROUGE
> [自动文摘评测方法：Rouge-1、Rouge-2、Rouge-L、Rouge-S](https://blog.csdn.net/qq_25222361/article/details/78694617) - CSDN博客 
- 一种机器翻译/自动摘要的评价准则

> [BLEU，ROUGE，METEOR，ROUGE-浅述自然语言处理机器翻译常用评价度量](https://blog.csdn.net/joshuaxx316/article/details/58696552) - CSDN博客 


# 语言模型

## 概率/统计语言模型 (PLM, SLM)
- **语言模型**是一种对语言打分的方法；而**概率语言模型**把语言的“得分”通过**概率**来体现
- 具体来说，概率语言模型计算的是**一个序列**作为一句话可能的概率
    ```
    Score("什么 是 语言 模型") --> 0.05   # 比较常见的说法，得分比较高
    Score("什么 有 语言 模型") --> 0.01   # 不太常见的说法，得分比较低
    ```
- 以上过程可以形式化为：
    <div style="width: 600px; margin: auto">![avater](eq6.PNG)</div>

  根据**贝叶斯公式**，有
    <div style="width: 600px; margin: auto">![avater](eq7.PNG)</div>

- 其中每个条件概率就是**模型的参数**；如果这个参数都是已知的，那么就能得到整个序列的概率了

### 参数的规模
- 设词表的大小为 `N`，考虑长度为 `T` 的句子，理论上有 `N^T` 种可能的句子，每个句子中有 `T` 个参数，那么参数的数量将达到 `O(T*N^T)`

### 可用的概率模型
- 统计语言模型实际上是一个概率模型，所以常见的概率模型都可以用于求解这些参数
- 常见的概率模型有：N-gram 模型、决策树、最大熵模型、隐马尔可夫模型、条件随机场、神经网络等
- 目前常用于语言模型的是 N-gram 模型和神经语言模型（下面介绍）


## N-gram 语言模型
- **马尔可夫(Markov)假设**——未来的事件，只取决于有限的历史
- 基于马尔可夫假设，N-gram 语言模型认为一个词出现的概率只与它前面的 n-1 个词相关
    <div style="width: 600px; margin: auto">![avater](eq8.PNG)</div>
  
- 根据**条件概率公式**与**大数定律**，当语料的规模足够大时，有
    <div style="width: 600px; margin: auto">![avater](eq9.PNG)</div>

- 以 `n=2` 即 bi-gram 为例，有
    <div style="width: 600px; margin: auto">![avater](eq10.PNG)</div>

- 假设词表的规模 `N=200000`（汉语的词汇量），模型参数与 n 的关系表  **注意这个的算法**
    <div style="width: 600px; margin: auto">![avater](eq11.PNG)</div>

### 可靠性与可区别性
- 假设没有计算和存储限制，`n` 是不是越大越好？
- 早期因为计算性能的限制，一般最大取到 `n=4`；如今，即使 `n>10` 也没有问题，
- 但是，随着 `n` 的增大，模型的性能增大却不显著，这里涉及了**可靠性与可区别性**的问题
- 参数越多，模型的可区别性越好，但是可靠性却在下降——因为语料的规模是有限的，导致 `count(W)` 的实例数量不够，从而降低了可靠性

### OOV 问题
- OOV 即 Out Of Vocabulary，也就是序列中出现了词表外词，或称为**未登录词**
- 或者说在测试集和验证集上出现了训练集中没有过的词
- 一般**解决方案**：
    - 设置一个词频阈值，只有高于该阈值的词才会加入词表
    - 所有低于阈值的词替换为 UNK（一个特殊符号）
- 无论是统计语言模型还是神经语言模型都是类似的处理方式
    > [NPLM 中的 OOV 问题](#nplm-中的-oov-问题)

### 平滑处理 TODO
- `count(W) = 0` 是怎么办？ 
- 平滑方法（层层递进）：
    - Add-one Smoothing (Laplace)
    - Add-k Smoothing (k<1)
    - Back-off （回退）
    - Interpolation （插值法）
    - Absolute Discounting （绝对折扣法）
    - Kneser-Ney Smoothing （KN）
    - Modified Kneser-Ney
    > [自然语言处理中N-Gram模型的Smoothing算法](https://blog.csdn.net/baimafujinji/article/details/51297802) - CSDN博客 


## 神经概率语言模型 (NPLM)
> [专题-词向量](./B-专题-词向量)
- 神经概率语言模型依然是一个概率语言模型，它通过**神经网络**来计算概率语言模型中每个参数
<div style="width: 600px; margin: auto">![avater](eq12.PNG)</div>
  
    - 其中 `g` 表示神经网络，`i_w` 为 `w` 在词表中的序号，`context(w)` 为 `w` 的上下文，`V_context` 为上下文构成的特征向量。
    - `V_context` 由上下文的**词向量**进一步组合而成

### N-gram 神经语言模型
> [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (Bengio, et al., 2003)
- 这是一个经典的神经概率语言模型，它沿用了 N-gram 模型中的思路，将 `w` 的前 `n-1` 个词作为 `w` 的上下文 `context(w)`，而 `V_context` 由这 `n-1` 个词的词向量拼接而成，即

<div style="width: 600px; margin: auto">![avater](eq13.PNG)</div>

    - 其中 `c(w)` 表示 `w` 的词向量
    - 不同的神经语言模型中 `context(w)` 可能不同，比如 Word2Vec 中的 CBOW 模型
- 每个训练样本是形如 `(context(w), w)` 的二元对，其中 `context(w)` 取 w 的前 `n-1` 个词；当不足 `n-1`，用特殊符号填充
    - **同一个网络只能训练特定的 `n`，不同的 `n` 需要训练不同的神经网络**

#### N-gram 神经语言模型的网络结构
- 【**输入层**】首先，将 `context(w)` 中的每个词映射为一个长为 `m` 的词向量，**词向量在训练开始时是随机的**，并**参与训练**；
- 【**投影层**】将所有上下文词向量**拼接**为一个长向量，作为 `w` 的特征向量，该向量的维度为 `m(n-1)`
- 【**隐藏层**】拼接后的向量会经过一个规模为 `h` 隐藏层，该隐层使用的激活函数为 `tanh`
- 【**输出层**】最后会经过一个规模为 `N` 的 Softmax 输出层，从而得到词表中每个词作为下一个词的概率分布
> 其中 `m, n, h` 为超参数，`N` 为词表大小，视训练集规模而定，也可以人为设置阈值
- 训练时，使用**交叉熵**作为损失函数
- **当训练完成时**，就得到了 N-gram 神经语言模型，以及副产品**词向量**
- 整个模型可以概括为如下公式：
   <div style="width: 600px; margin: auto">![avater](eq14.PNG)</div>

### 模型参数的规模与运算量
- 模型的超参数：`m, n, h, N`
    - `m` 为词向量的维度，通常在 `10^1 ~ 10^2`
    - `n` 为 n-gram 的规模，一般小于 5
    - `h` 为隐藏的单元数，一般在 `10^2`
    - `N` 位词表的数量，一般在 `10^4 ~ 10^5`，甚至 `10^6`
- 网络参数包括两部分
    - 词向量 `C`: 一个 `N * m` 的矩阵——其中 `N` 为词表大小，`m` 为词向量的维度 
    - 网络参数 `W, U, p, q`：
        ```
        - W: h * m(n-1) 的矩阵
        - p: h * 1      的矩阵
        - U: N * h    的矩阵
        - q: N * 1    的矩阵
        ```
- 模型的运算量
    - 主要集中在隐藏层和输出层的矩阵运算以及 SoftMax 的归一化计算
    - 此后的相关研究中，主要是针对这一部分进行优化，其中就包括 **Word2Vec** 的工作

### 相比 N-gram 模型，NPLM 的优势
- 单词之间的相似性可以通过词向量来体现
    > 相比神经语言模型本身，作为其副产品的词向量反而是更大的惊喜
    >
    > [词向量的理解](./B-专题-词向量#词向量的理解)
- 自带**平滑**处理

### NPLM 中的 OOV 问题
- 在处理语料阶段，与 N-gram 中的处理方式是一样的——将不满阈值的词全部替换为 UNK
**神经网络**中，一般有如下几种处理 UNK 的思路
- 为 UNK 分配一个随机初始化的 embedding，并**参与训练**
    > 最终得到的 embedding 会有一定的语义信息，但具体好坏未知
- 把 UNK 都初始化成 0 向量，**不参与训练**
    > UNK 共享相同的语义信息
- 每次都把 UNK 初始化成一个新的随机向量，**不参与训练**
    > 常用的方法——因为本身每个 UNK 都不同，随机更符合对 UNK 基于最大熵的估计
    >> [How to add new embeddings for unknown words in Tensorflow (training & pre-set for testing)](https://stackoverflow.com/questions/45113130/how-to-add-new-embeddings-for-unknown-words-in-tensorflow-training-pre-set-fo) - Stack Overflow
    >>
    >> [Initializing Out of Vocabulary (OOV) tokens](https://stackoverflow.com/questions/45495190/initializing-out-of-vocabulary-oov-tokens) - Stack Overflow
- 基于 Char-Level 的方法
    > PaperWeekly 第七期 -- [基于Char-level的NMT OOV解决方案](https://zhuanlan.zhihu.com/p/22700538?refer=paperweekly) 




# 句嵌入
===
参考：https://github.com/ava-YangL/Algorithm_Interview_Notes-Chinese/blob/master/B-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/B-%E4%B8%93%E9%A2%98-%E5%8F%A5%E5%B5%8C%E5%85%A5.md
Index
---
<!-- TOC -->

- [基线模型](#基线模型)
    - [基于统计的词袋模型（BoW）](#基于统计的词袋模型bow)
    - [基于词向量的词袋模型](#基于词向量的词袋模型)
        - [均值模型](#均值模型)
        - [加权模型](#加权模型)
    - [基于 RNN（任务相关）](#基于-rnn任务相关)
    - [基于 CNN（任务相关）](#基于-cnn任务相关)
- [词袋模型](#词袋模型)
    - [[2018] Power Mean 均值模型](#2018-power-mean-均值模型)
    - [[2017] SIF 加权模型](#2017-sif-加权模型)
    - [[]](#)
- [无监督模型](#无监督模型)
    - [[2015] Skip-Thought Vector](#2015-skip-thought-vector)
    - [[2018] Quick-Thought Vectors](#2018-quick-thought-vectors)
- [有监督模型](#有监督模型)
    - [[2017] InferSent](#2017-infersent)
    - [[2017] Self-Attention](#2017-self-attention)
    - [[2015] DAN & RecNN](#2015-dan--recnn)
- [多任务学习](#多任务学习)
    - [[2018] 基于多任务的 Sentence Embedding（微软）](#2018-基于多任务的-sentence-embedding微软)
    - [[2018] Universal Sentence Encoder（谷歌）](#2018-universal-sentence-encoder谷歌)
- [参考文献](#参考文献)

<!-- /TOC -->


## 基线模型

### 基于统计的词袋模型（BoW）
- 单个词的 One-Hot 表示
- 基于频数的词袋模型
- 基于 TF-IDF 的词袋模型  : TF-IDF（term frequency–inverse document frequency），可见 https://zhuanlan.zhihu.com/p/31197209  
- ... 

> 词袋模型（英語：Bag-of-words model）是個在自然語言處理和信息檢索(IR)下被簡化的表達模型。 此模型下，一段文本（比如一个句子或是一个文档）可以用一個装着这些词的袋子来表示，這種表示方式不考慮文法以及詞的順序。 最近词袋模型也被應用在電腦視覺領域。

### 基于词向量的词袋模型
#### 均值模型
  <div style="width: 600px; margin: auto">![avater](eq15.PNG)</div>
> 其中 `v_i` 表示维度为 `d` 的词向量，均值指的是对所有词向量**按位求和**后计算每一维的均值，最后 `s` 的维度与 `v` 相同。

#### 加权模型
<div style="width: 600px; margin: auto">![avater](eq16.PNG)</div>
> 其中 `α` 可以有不同的选择，但一般应该遵循这样一个准则：**越常见的词权重越小**
>> [[2017] SIF 加权模型](#2017-sif-加权模型)

### 基于 RNN（任务相关）
- 以最后一个隐状态作为整个句子的 Embedding
- 基于 RNN 的 Sentence Embedding 往往用于特定的有监督任务中，**缺乏可迁移性**，在新的任务中需要重新训练；
- 此外，由于 RNN 难以并行训练的缺陷，导致开销较大。 (难以并行化是因为RNN的顺序关系？？卷积神经网路比较容易并行化？)


### 基于 CNN（任务相关）
- 卷积的优势在于提取**局部特征**，利用 CNN 可以提取句子中类似 n-gram 的局部信息；
- 通过整合不同大小的 n-gram 特征作为整个句子的表示。

   <div style="width: 800px; margin: auto">![avater](eq17.PNG)</div>

## 词袋模型

### [2018] Power Mean 均值模型
> [4]
- 幂均值
<div style="width: 800px; margin: auto">![avater](eq18.PNG)</div>


### [2017] SIF 加权模型
- 频率高的词权重小
- 主成分 （独立性？）
<div style="width: 800px; margin: auto">![avater](eq19.PNG)</div>



## 无监督模型
这个没认真看
<div style="width: 800px; margin: auto">![avater](eq20.PNG)</div>


## 有监督模型

### [2017] InferSent
> [5]
本文使用有监督的方法，在自然语言推理（NLI）数据集上训练 Sentence Embedding；
本文认为从 NLI 数据集（比如 SNLI）中训练得到的句向量也适合迁移到其他 NLP 任务中。
就像在各种 CV 任务中使用基于 ImageNet 的模型（VGG, ResNet 等）来得到图像特征一样，在处理 NLP 任务之前可以先使用本文公开的模型来计算句子的特征。

### [2017] Self-Attention
> [3]
本文提出使用二维矩阵作为句子表征，矩阵的行表示在句子不同位置的关注度，以解决句子被压缩成一维向量时的信息损失。

### [2015] DAN & RecNN
> [9]

## 多任务学习
- InferSent 模型的成功，使大家开始探索不同的有监督任务中得到的 Sentence Embedding 在下游任务中的效果。
- 多任务学习试图在一次训练中组合不同的训练目标。

### [2018] 基于多任务的 Sentence Embedding（微软）
> [6]

- 本文认为为了能够推广到各种不同的任务，需要对同一句话的多个方面进行编码。
- 简单来说，模型同时在**多个任务**和**多个数据源**上进行训练，但是**共享**相同的 Sentence Embedding。
- 任务及数据集包括：
    - Skip-Thought（预测上一句/下一句）——BookCorpus
    - 神经机器翻译（NMT）——En-Fr (WMT14) + En-De (WMT15)
    - 自然语言推理（NLI）——SNLI + MultiNLI
    - Constituency Parsing——PTB + 1-billion word

- 本文模型与 [Skip-Thought Vector](#2015-skip-thought-vector) 基本一致
    - **主要区别**在于本文的 Encoder 部分使用的是 **Bi-GRU**，而 Decoder 部分完全一致；
    - 使用 GRU 而非 LSTM 的原因主要是为了速度；

### [2018] Universal Sentence Encoder（谷歌）
> [7]

- 本文的目的是动态地适应各种的 NLP 任务，通过在不同的数据集和不同的任务上同时训练。
- 本文使用**类似的多任务框架**，区别在于使用的 Encoder 不同。
    > [[2018] 基于多任务的 Sentence Embedding（微软）](#2018-基于多任务的-sentence-embedding微软)
- 本文以两种模型作为 Encoder
    - **Transformer** [8]——更高的精度
    - **DAN** (Deep Averaging Network) [9]——更快的速度
  
- 一个可用的预训练版本
    ```python
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    embeddings = embed([
        "The quick brown fox jumps over the lazy dog.",
        "I am a sentence for which I would like to get its embedding"])

    sess.run(embeddings)
    ```
    > TensorFlow Hub  |  [universal-sentence-encoder](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/2) 


## 参考文献
- [1] A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2016.
- [2] Skip-Thought Vectors, NIPS 2015.
- [3] A Structured Self-attentive Sentence Embedding, ICLR 2017.
- [4] An efficient framework for learning sentence representations, ICLR 2018.
- [5] Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, ACL 2017.
- [6] Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, ICLR 2018.
- [7] Universal Sentence Encoder, arXiv 2018.
- [8] Attention is all you need, NIPS 2017.
- [9] Deep unordered composition rivals syntactic methods for text classification, 2015 ACL.