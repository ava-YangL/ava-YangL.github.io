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

方法：
（1）获得训练好的模型参数：隐藏状态+观测符号+状态转移概率+状态随观测符号的发射概率
- 标注了的话很好算
- 没标注的话，通过一些辅助资源，利用前向-后向算法学习一个hmm
（2）维特比算法 
- 根据模型和观测序列，确定隐藏状态

<div style="width: 600px; margin: auto">![avater](1.jpg)</div>
可以参见： https://github.com/IdearHui/posTag