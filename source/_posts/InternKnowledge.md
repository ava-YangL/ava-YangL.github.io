---
title: InternKnowledge
date: 2019-09-05 14:48:22
tags:
---

整理一下Intern期间的知识点。
<!--more-->

#### 1 多线程


##### 堆栈
堆：　是大家共有的空间，分**全局堆**和**局部堆**。全局堆就是所有没有分配的空间，局部堆就是用户分配的空间。**堆**在操作系统对**进程初始化**的时候分配，运行过程中也可以向系统要额外的堆，但是记得用完了要**还**给操作系统，要不然就是**内存泄漏**。

栈：是个**线程**独有的，保存其**运行状态**和**局部自动变量**的。**栈**在**线程开始**的时候初始化，每个线程的**栈互相独立**，因此，栈是　thread safe的。每个Ｃ＋＋对象的数据成员也存在在栈中，**每个函数都有自己的栈**，栈被用来在函数之间**传递参数**。操作系统在**切换线程的时候会自动的切换栈**，就是切换**ＳＳ／ＥＳＰ寄存器**。栈空间不需要在高级语言里面显式的分配和释放。

#####　什么是线程安全？
如果多线程的程序运行结果是可预期的，而且与单线程的程序运行结果一样，那么说明是“线程安全”的。

##### 多线程同步和互斥有何异同，在什么情况下分别使用他们？举例说明
- 所谓同步，表示有先有后，比较正式的解释是“线程同步是指线程之间所具有的一种制约关系，一个线程的执行依赖另一个线程的消息，当它没有得到另一个线程的消息时应等待，直到消息到达时才被唤醒。”
- 所谓互斥，比较正式的说明是“线程互斥是指对于共享的进程系统资源，在各单个线程访问时的排它性。当有若干个线程都要使用某一共享资源时，任何时刻最多只允许一个线程去使用，其它要使用该资源的线程必须等待，直到占用资源者释放该资源。线程互斥可以看成是一种特殊的线程同步。”表示不能同时访问，也是个顺序问题，所以互斥是一种特殊的同步操作。
- 举个例子，设有一个全局变量global，为了保证线程安全，我们规定只有当主线程修改了global之后下一个子线程才能访问global，这就需要同步主线程与子线程，可用关键段实现。当一个子线程访问global的时候另一个线程不能访问global，那么就需要互斥。

##### 线程的基本概念、线程的基本状态及状态之间的关系？

概念：线程是进程中执行运算的最小单位，是进程中的一个实体，是被系统独立调度和分派的基本单位，线程自己不拥有系统资源，只拥有一点在运行中必不可少的资源，但它可与同属一个进程的其它线程共享进程所拥有的全部资源。一个线程可以创建和撤消另一个线程，同一进程中的多个线程之间可以并发执行。

  好处 ：

（1）易于调度。

               （2）提高并发性。通过线程可方便有效地实现并发性。进程可创建多个线程来执行同一程序的不同部分。

               （3）开销少。创建线程比创建进程要快，所需开销很少。。

               （4）利于充分发挥多处理器的功能。通过创建多线程进程，每个线程在一个处理器上运行，从而实现应用程序的并发性，使每个处理器都得到充分运行。

状态：运行、阻塞、挂起阻塞、就绪、挂起就绪

    状态之间的转换：准备就绪的进程，被CPU调度执行，变成运行态；

                                 运行中的进程，进行I/O请求或者不能得到所请求的资源，变成阻塞态；

                                 运行中的进程，进程执行完毕（或时间片已到），变成就绪态；

                                 将阻塞态的进程挂起，变成挂起阻塞态，当导致进程阻塞的I/O操作在用户重启进程前完成（称之为唤醒），挂起阻塞态变成挂起就绪态，当用户在I/O操作结束之前重启进程，挂起阻塞态变成阻塞态；

                                 将就绪（或运行）中的进程挂起，变成挂起就绪态，当该进程恢复之后，挂起就绪态变成就绪态





Ref：
- https://blog.csdn.net/Primeprime/article/details/79080015

#### 2 RNN LSTM GRU


#### 3 Transformer
优点：
- 全连接和Attention的结合体。
- 将任意两个单词的距离是1 这对解决NLP中棘手的长期依赖问题是非常有效的。  (因为不是RNN那种顺序依赖，所以是一个句话作为矩阵输入进去的。所以单词的距离是1 )
- 并行性体现在哪儿？ 符合目前的硬件（主要指GPU）环境。
- 抛弃了CNN RNN
缺点：
- （1）粗暴的抛弃RNN和CNN虽然非常炫技，但是它也使模型丧失了捕捉局部特征的能力，RNN + CNN + Transformer的结合可能会带来更好的效果。
- （2）Transformer失去的位置信息其实在NLP中非常重要，而论文中在特征向量中加入Position Embedding也只是一个权宜之计，并没有改变Transformer结构上的固有缺陷。



Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成，更准确地讲，Transformer由且仅由**self-Attenion**和**Feed Forward Neural Network**组成 ，一个基于Transformer的可训练的神经网络可以通过**堆叠Transformer**的形式进行搭建。

Attention机制原因：考虑到RNN（或者LSTM，GRU等）的计算限制为是**顺序**的，也就是说RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：
- 时间片 t 的计算依赖 t-1 时刻的计算结果，这样**限制了模型的并行能力**；
- 顺序计算的过程中信息会丢失，尽管**LSTM等门机制**的结构一定程度上缓解了**长期依赖**的问题，但是对于特别长期的依赖现象,LSTM依旧无能为力。

那么Transformer怎么解决的呢：
- Attention机制，将序列中的任意两个位置之间的**距离**是缩小为一个**常量**；？？
- 不是顺序结构，是并行结构，更符合现有的gpu框架。

Ref：
- https://zhuanlan.zhihu.com/p/48508221
-    （推荐看）

##### (1) Transformer结构

本质是个Encoder-Decoder
<div style="width: 800px; margin: auto">![avater](1.PNG)</div>
每个Encoder/decoder的结构：
<div style="width: 800px; margin: auto">![avater](2.PNG)</div>
Encoder：
<div style="width: 800px; margin: auto">![avater](3.PNG)</div>
Decoder和encoder的不同之处在于Decoder多了一个**Encoder-Decoder Attention**，两个Attention分别用于计算输入和输出的权值：
- Self-Attention：当前翻译和已经翻译的**前文**之间的关系；(以机器翻译来讲)
- Encoder-Decnoder Attention：当前翻译和**编码**的特征向量之间的关系。

##### (2) 输入编码
首先通过Word2Vec等词嵌入方法将输入语料转化成特征向量，论文中使用的词嵌入的维度为 500 。在最底层的block中， x将直接作为Transformer的输入，而在其他层中，输入则是上一个block的输出。为了画图更简单，我们使用更简单的例子来表示接下来的过程
<div style="width: 800px; margin: auto">![avater](4.PNG)</div>

##### (3)  Self Attention
> 注意呀，这里以句子为单位输入，输入每个词的embeddings

**Self Attention介绍**
> The animal didn't cross the street because it was too tired
> 核心内容是为输入向量的每个单词学习一个权重，例如在下面的例子中我们判断it代指的内容，

在self-attention中，每个单词有3个不同的向量，它们分别是Query向量（ Q ），Key向量（ K ）和Value向量（ V ），长度均是64。它们是由嵌入向量 X 乘以三个不同的权值矩阵 $W^Q W^K W^V$ 得到，其中三个矩阵的尺寸也是相同的。均是 512×64 。
<div style="width: 800px; margin: auto">![avater](5.PNG)</div>
上述步骤转化为下图：
<div style="width: 800px; margin: auto">![avater](6.PNG)</div>
实际计算过程中是采用基于矩阵的计算方式，那么论文中的 Q V K 的计算方式如图：
<div style="width: 800px; margin: auto">![avater](7.PNG)</div>
那么总结self attention的过程就是：
<div style="width: 800px; margin: auto">![avater](8.PNG)</div>
在self-attention需要强调的最后一点是其采用了残差网络 [5]中的short-cut结构，目的当然是解决深度学习中的退化问题，得到的最终结果如图。
<div style="width: 800px; margin: auto">![avater](9.PNG)</div>

> 注意，qi x ki 得到的是 单词 xi 的权重， 那么每个单词都能得到这样一个权重， softmax， 这样就是 这句话的每次单词加权代表本个单词。 最后还是每个单词有个 zi的。

##### (4)  Multi Head Attention
<div style="width: 800px; margin: auto">![avater](10.PNG)</div>

##### (5)  Encoder-Decoder Attention
在解码器中，Transformer block比编码器中多了个encoder-cecoder attention。在encoder-decoder attention中， **Q 来之与解码器的上一个输出**， K 和 V则来自于与编码器的输出。其计算方式完全和图10的过程相同。

由于在机器翻译中，解码过程是一个顺序操作的过程，也就是当解码第 k 个特征向量时，我们只能看到第 k-1 及其之前的解码结果，论文中把这种情况下的multi-head attention叫做masked multi-head attention。

##### (6) 损失层
<div style="width: 800px; margin: auto">![avater](11.PNG)</div>

解码器解码之后，解码的特征向量经过一层激活函数为softmax的全连接层之后得到反映每个单词概率的输出向量。此时我们便可以通过CTC等损失函数训练模型了

##### (7) 位置编码
截止目前为止，我们介绍的Transformer模型并没有捕捉顺序序列的能力，也就是说无论句子的结构怎么打乱，Transformer都会得到类似的结果。换句话说，Transformer只是一个功能更强大的词袋模型而已。

论文中在编码词向量时引入了位置编码（Position Embedding）的特征。具体地说，位置编码会在词向量中加入了单词的位置信息，这样Transformer就能区分不同位置的单词了。

那么怎么编码这个位置信息呢？常见的模式有：a. 根据数据学习；b. 自己设计编码规则。在这里作者采用了第二种方式。那么这个位置编码该是什么样子呢？通常位置编码是一个长度为 d_model 的特征向量，这样便于和词向量进行单位加的操作，如图

<div style="width: 800px; margin: auto">![avater](12.PNG)</div>



#### 4 Bert
- Paper Link ： https://arxiv.org/pdf/1810.04805.pdf
- Team： Google AI

Bidirectional Encoder Representations from Transformers。
Based on Finetune, 预训练可以改进许多NLP任务。
目前将 pretrained embeddings 应用于下游任务存在两种策略：基于特征的策略和微调策略（fine-tuning）。
- 基于特征的策略（如 ELMo）使用将预训练表征作为**额外特征**的任务专用架构。
- 微调策略（如生成预训练 Transformer (OpenAI GPT)）引入了任务特定最小参数，通过简单地微调预训练参数在下游任务中进行训练。在之前的研究中，两种策略在预训练期间使用相同的目标函数，利用单向语言模型来学习通用语言表征。


创新点：
- 1 改进了基于Fine Tune的策略， masked language model，MLM， 克服OPEN AI GPT的单向缺陷。MLM结合了左右的语境，预测被mask的词id，就可以pretrain一个深度双向的Tranformer。
    - 注意独立训练的从左到右和从右到左的LM浅层级联，其实是没有这种效果好的。
- 2 Next sentence prediction

贡献：
- Finetune+下游任务，不那么需要精心设计任务特定的架构了。他的性能比许多使用特定架构的还好。

架构：
- 多层双向 Transformer 编码器？
- 


疑问：
- Transformer



Ref:
- https://www.jiqizhixin.com/articles/2018-10-12-13





#### 3 ELMO


#### 4 OPENAI GPT
从左到右的架构，其中每个 token 只能注意 Transformer 自注意力层中的先前 token。






#### 3 NLP 任务
- 自然语言推断
- 复述，paraphrasing
- 命名实体识别，NER
- SQuAD问答
- GLUE?
- MultiNLI?
- 机器翻译，BLEU?


HMM


CRF


NLTK



FLAIR
