# reinforcement learning in nlp: A survey

强化学习在nlp领域的应用主要有对话管理和文本生成两个方面。Q-Learning [34] is a popular form of RL. This model-free technique is used to learn an optimal action-value function Q(s, a)


http://blog.dennybritz.com/2015/09/11/reimagining-language-learning-with-nlp-and-reinforcement-learning/

 - [Language Understanding for Text-based Games using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf)

论文为text-based games学习控制策略，采用deep reinforcement learning架构来学习状态表示和动作策略，游戏的rewards作为反馈。这个架构使我们可以将
文本映射为向量表征，这样会捕捉游戏状态的语义。

选择状态表示：bag of words忽略词顺序。本文采用reinforcement learning架构，将游戏序列组成MDP，智能体学习一个策略，这个策略是以action-value函数Q(s,a)
的形式表示，Q函数表示长期的收益，用deep rnn网络来拟合Q函数，rnn有两个模块，第一个将文本转化为向量表示，这部分是用LSTM执行的；第二个模块
在给定第一个计算出来的向量表示的情况下，对动作评分。

 - [On-line Active Reward Learning for Policy Optimisation in Spoken Dialogue Systems](https://arxiv.org/pdf/1605.07669v2.pdf)

在强化学习的研究中，如何定义和学习奖励机制（reward）是十分重要的，然而，传统的用户反馈的方法开销很大，在实用系统中无法承受。文章提出一种在线学习的框架，首先通过循环神经网络生成对话表示，进而利用基于高斯过程的主动学习机制大大降低了人工标注的代价。University of Cambridge这个研究组在对话系统方面有着长期深入的研究工作，建议感兴趣的同学可以多关注这方面的工作。


 - [Deep Reinforcement Learning with a Natural Language Action Space](https://arxiv.org/pdf/1511.04636v5.pdf)

摘要：这篇文章介绍了一个新的带有深度神经网络的强化学习架构，来处理由自然语言描述的state和action空间，这个在text-based games中也出现了。
DRRN，这个架构用独立的词向量表示行动和状态空间，词向量与interaction function结合来估计Q-function。
实验：在两个text games中评估DRRN，性能比其他DQN架构要好，并且在paraphrased action descriptions实验中，这个模型不仅能记忆text string，
还可以提取意义。

这篇文章关注序列决策任务的学习策略，用自然语言描述强化学习中的state和action。
## 2.[EMNLP 2016](http://blog.aylien.com/highlights-emnlp-2016-dialogue-deeplearning-and-more/)
 - [Modeling Human Reading with Neural Attention](https://arxiv.org/pdf/1608.05604.pdf)
 
这篇文章是关于阅读理解。人类阅读文本就是通过做一系列的注视和扫视的行为，本文用一个非监督架构模拟人阅读时skipping和reading的行为，这个架构将自编码机与神经注意力模型结合，并用强化学习训练。这个模型将人类阅读行为解释为语言理解精度（将输入的词语正确编码）与注意力的经济性（尽量少的注视词语）之间的tradeoff。已经有很多模型解析人类阅读时eye-movements，但更多的是监督方法用eye-tracking data训练模型。本文提出的新的模型架构可以解释哪些词被跳过，哪些词被看到，以及预测被看到词的阅读次数，并且方法是无监督，只需要非标注数据。

 - [Language as a Latent Variable:Discrete Generative Models for Sentence Compression](https://arxiv.org/pdf/1609.07317.pdf)
这篇文章用于compressing sentence。文本的深度生成模型，
Andriy Mnih和Volodymyr Mnih的文章[1]()[2]()

## 3.[ACL 2016 Highlights]

## 4.[SIGDIAL 2016 Highlights]

## 5.[NIPS,ICML]

