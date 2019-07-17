---
title: POS tagging
date: 2019-07-16 14:41:06
tags:
---


#### 1 HMM

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




#### 2 CRF

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