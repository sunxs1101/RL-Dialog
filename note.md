# reinforcement learning in nlp: A survey

强化学习在nlp领域的应用主要有对话管理和文本生成两个方面。Q-Learning [34] is a popular form of RL. This model-free technique is used to learn an optimal action-value function Q(s, a)


http://blog.dennybritz.com/2015/09/11/reimagining-language-learning-with-nlp-and-reinforcement-learning/

 - [Language Understanding for Text-based Games using Deep Reinforcement Learning](http://people.csail.mit.edu/karthikn/pdfs/mud-play15.pdf)

论文为text-based games学习控制策略，采用deep reinforcement learning架构来学习状态表示和动作策略，游戏的rewards作为反馈。这个架构使我们可以将
文本映射为向量表征，这样会捕捉游戏状态的语义。本文采用reinforcement learning架构，将游戏序列组成MDP，智能体学习一个策略，这个策略是以action-value函数Q(s,a)的形式表示，Q函数表示长期的收益，用deep rnn网络来拟合Q函数，rnn有两个模块，第一个将文本转化为向量表示，这部分是用LSTM执行的；第二个模块在给定第一个计算出来的向量表示的情况下，对动作评分。

选择状态表示：bag of words忽略词顺序。

 - [On-line Active Reward Learning for Policy Optimisation in Spoken Dialogue Systems](https://arxiv.org/pdf/1605.07669v2.pdf)

这篇是ACL2016学生Best Paper。在强化学习的研究中，如何定义和学习奖励机制（reward）是十分重要的，然而，传统的用户反馈的方法开销很大，在实用系统中无法承受。文章提出一种在线学习的框架，首先通过循环神经网络生成对话表示，进而利用基于高斯过程的主动学习机制大大降低了人工标注的代价。
摘要：计算精确的reward function对于通过强化学习来优化一个对话策略很重要。实际应用中，用显式的用户反馈作为reward信号往往不可靠并且难收集。如果用户的intent提前知道，或者数据可获取用于离线预训练，但实际上这两个都不能应用于大多数实际系统的应用。这里我们提出了一个在线学习架构，对话策略通过基于高斯过程的主动学习机制来训练，高斯过程在一个用rnn encoder-decoder生成的连续空间对话表示中作用。实验表明提出的这个架构能够显著降低数据注释成本和噪声用户反馈。

通过Gaussian过程分类法与一种基于神经网络的无监管式对话嵌入方法，提出了一种主动的奖赏函数学习模型。通过Reinforcement Learning的方式，主动询问客户收集更多的信息来得到精确的奖赏函数，实现在线学习策略。相比人工语聊标注以及离线模型的方式上有很大的创新。 


 - [Deep Reinforcement Learning with a Natural Language Action Space](https://arxiv.org/pdf/1511.04636v5.pdf)

摘要：这篇文章介绍了一个新的带有深度神经网络的强化学习架构DRRN，来处理由自然语言描述的state和action空间，学习序列决策任务的策略。DRRN这个架构用独立的词向量表示行动和状态空间，词向量与interaction function结合来估计Q-function。
实验：在两个text games中评估DRRN，性能比其他DQN架构要好，并且在paraphrased action descriptions实验中，这个模型不仅能记忆text string，
还可以提取意义。

这篇文章关注用自然语言描述强化学习中的state和action。
## 2.[EMNLP 2016](http://blog.aylien.com/highlights-emnlp-2016-dialogue-deeplearning-and-more/)
 - [Modeling Human Reading with Neural Attention](https://arxiv.org/pdf/1608.05604.pdf)
 
这篇文章是关于阅读理解。人类阅读文本就是通过做一系列的注视和扫视的行为，本文用一个非监督架构模拟人阅读时skipping和reading的行为，这个架构将自编码机与神经注意力模型结合，并用强化学习训练。这个模型将人类阅读行为解释为语言理解精度（将输入的词语正确编码）与注意力的经济性（尽量少的注视词语）之间的tradeoff。已经有很多模型解析人类阅读时eye-movements，但更多的是监督方法用eye-tracking data训练模型。本文提出的新的模型架构可以解释哪些词被跳过，哪些词被看到，以及预测被看到词的阅读次数，并且方法是无监督，只需要非标注数据。

 - [Language as a Latent Variable:Discrete Generative Models for Sentence Compression](https://arxiv.org/pdf/1609.07317.pdf)

这篇文章用于compressing sentence。文章采用文本的深度生成模型，文档的隐语义表示从离散语言模型分布中提取，建立变分自编码机用于模型中的推断，然后应用与compressing sentence任务。在应用中，生成模型首先从语言模型中提取latent summary sentence，接着在latent summary中提取observed sentence。

seq2seq框架在nlg中取得很大成功，也发展出attention这种机制，但这些都是判别模型，训练后用于估计条件输出分布。这篇文章用深度生成模型计算联合概率分布，采用离散变分自编码机(VAE)用于推断。VAE依赖reparameterisation trick，并不适用于离散隐语言模型，这里采用REINFORCE算法减轻基于采样的变分推断存在的高方差问题。文章引用了两篇REINFORCE算法的文章，分别是Andriy Mnih的[Neural Variational Inference and Learning in Belief Networks](https://arxiv.org/pdf/1402.0030v2.pdf)和Volodymyr Mnih的文章[Recurrent Models of Visual Attention](https://arxiv.org/pdf/1406.6247v1.pdf)，这两篇文章用梯度来训练推断网络，这可以看做REINFORCE算法的一个特例。REINFORCE算法最早在[Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://webdocs.cs.ualberta.ca/~sutton/williams-92.pdf)提出，它是一类联合的reinforcement learning算法，用于有stochastic units的连接网络。

插一句：[nips2016](http://it.sohu.com/20161213/n475750438.shtml)上David Blei 深入介绍了变分推理（Variational Inference）研究的最近的多项进展。最有影响的还是重新参数化（reparameterization）的技巧，该技巧可以通过随机变量实现反向传播，同时也推动了变自编码器上最新进展。

 - [Deep Reinforcement Learning with a Combinatorial Action Space for Predicting Popular Reddit Threads](https://arxiv.org/pdf/1606.03667.pdf)
 
本文是[Deep Reinforcement Learning with a Natural Language Action Space](https://arxiv.org/pdf/1511.04636v5.pdf)延续，用deep reinforcement learning架构做online popularity prediction和tracking，采用Reddit的数据，两方面贡献：一，新的强化学习任务，状态和动作由自然语言定义；二，提出新的deep reinforcement learning架构DRRN-Sum和DRRN-BiLSTM用于解决与自然语言关联的动作空间，效果比DRRN,per-action DQN和其他处理自然语言动作空间的DQN变种要好。

## 3.[ACL 2016 Highlights]

### ACL2016优秀论文解读

## 4.[NIPS]

机器学习方法可以分为生成方法（generative approach）和判别方法（discriminative approach），所学到的模型分别称为生成式模型（generative model）和判别式模型（discriminative model）。其中近两年来流行的生成式模型主要分为三种方法：

 - 生成对抗网络（GAN：Generative Adversarial Networks） 
 - 变分自编码器（VAE: Variational Autoencoders） 
 - 自回归模型（Autoregressive models） 

#### [DeepMind NIPS 2016论文盘点Part1](http://it.sohu.com/20161203/n474811175.shtml)

 - [safe and efficient off-policy reinforcement learning](https://arxiv.org/abs/1606.02647)

我们的目标是设计出带有两个所需特性的强化学习算法。首先，要使用离策略数据（off-policy data），当我们使用记忆再现（memory replay，即观察日志数据）时它对探索很重要。其次，要使用多步骤返回（multi-steps returns）以便更快地传递反馈和避免近似/估计误差（approximation/estimation errors）的积累。这两个属性在深度强化学习中是至关重要的。

 - [Deep Exploration via Bootstrapped DQN](https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn)

在复杂环境中进行有效的探索对强化学习（RL）来说仍然是一个巨大的挑战。最近我们看到在强化学习领域出现了很多突破，但这些算法中很多都需要巨量的数据（数百万次博弈），之后才能学会做出好的决策。在许多真实世界环境中，我们是无法获得这样大量的数据的。

这些算法学习如此之慢的原因之一是它们并没有收集到用于学习该问题的「正确（right）」数据。这些算法使用抖动（dithering）（采取随机动作）来探索他们的环境——这种方法比起在多个时间步骤上对信息策略进行优先级排序的深度探索（deep exploration），效率指数级地更差。对于使用深度探索进行统计学上高效的强化学习已经有很多文献了，但问题是这些算法中没有一个是可以通过深度学习解决的……而现在我们有了。

这篇论文的关键突破如下：

 - 我们提出了第一个结合了深度学习与深度探索的实用强化学习算法：Bootstrapped DQN。
 - 我们表明这个算法可以给学习速度带来指数级的提升。
 - 我们展示了在 Atari 2600 上的当前最佳的结果。

视频连接：https://www.youtube.com/playlist?list=PLdy8eRAW78uLDPNo1jRv8jdTx7aup1ujM
#### [part2](http://it.sohu.com/20161207/n475166788.shtml)
#### [nips其它有价值的论文](http://it.sohu.com/20161213/n475750438.shtml)
 - Value Iteration Network
令人印象深刻：该论文的主要创新在于其模型包含了一个可微分的「规划模块（planning module），这让网络可以做出规划并更好地泛化到其它从未见过的领域。

 - [Sequential Neural Models with Stochastic Layers 以及 Phased LSTMs](http://ulrichpaquet.com/Papers/1605.07571v1.pdf)
前者将 状态空间模型（State Space Model）的想法和 RNN 结合起来，充分利用了两个领域的最好的东西。后者将「time gate」添加到了 LSTM 中，这显著改善了针对长序列数据的优化和表现。

- 通过对抗训练生成文本（Generating Text via Adversarial Training）、用于带有 Gumbel-softmax 分布的离散元素的序列的 GAN（GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution）、对话模型的对抗式评估（Adversarial Evaluation of Dialogue Models）。短评：我对模拟和评估对话系统的技术很感兴趣。

- 构建像人一样学习和思考的机器（Building Machines That Learn and Think Like People）。短评：这个主题演讲非常好，所以我想要深入了解一下论文。这个演讲探索了人类利用大量先验知识的方式，以及我们可以如何将其整合进我们的系统中；其中一些特定的观察结果为我们带来了一些可以执行的研究方向。（这似乎和对话有关，因为这个研究可能能够解释类似「the blorf flazzed the peezul」这样的无意义陈述的伪可理解性（pseudo-intelligibility）。 

- 跨许多个数量级学习价值（Learning values across many orders of magnitude）。短评：粗略看这可能是关于优化（optimization）的，但在反事实的背景（counterfactual setups）中，这个问题是很普遍的。我可是很喜欢把规模不变性用作一个有用的先验知识（scale invariance as a useful prior）。

- 用于神经结构预测的回报增强最大似然（Reward Augmented Maximum Likelihood for Neural Structured Prediction）短评：这可以被看作是另一种使用世界的模型来转移强化学习的样本复杂性的方法。（比如：如果编辑距离（edit distance）只是该回报的初始模型呢？

- 安全高效的离策略强化学习（Safe and Efficient Off-Policy Reinforcement Learning）。短评：这是一个重要的设置。这种特别的调整让人联想到了之前这一领域提出的估计器（estimator，参阅论文《Learning from Logged Implicit Exploration Data》）；但尽管如此，这还是很有意思。

- Improved dropout for shallow deep learning：提出一种改进版本dropout 
链接：https://www.youtube.com/watch?v=oZOOfaT94iU&feature=youtu.be

## 4.[SIGDIAL 2016 Highlights]

## 5.[ICLR 2017](http://it.sohu.com/20161204/n474874145.shtml)
[ICLR 2017 有什么值得关注的亮点？](https://www.zhihu.com/question/52311422/answer/130508707)

 - [GENERATING LONG AND DIVERSE RESPONSES WITH NEURAL CONVERSATION MODELS](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/8/83/Dedaff23ad393c48fe7b7989542318a02dc0a06e.pdf)
 - [Learning to compose words into sentences with reinforcement learning]()
 
 - [NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://openreview.net/pdf?id=r1Ue8Hcxg)
来自Google Brain. 利用RL来优化RNN的结构。神经网络不容易设计，需要很多专业知识。本文用RNN生成神经网络的模型描述，用强化学习训练这个RNN来最大化生成的网络结构的准确率。在一些数据集上，会比现有的state-of-the-art model，如LSTM要好。
论文提出神经结构搜索，一个基于梯度的方法来寻找好的结构。准确率作为reward signal，计算policy gradient来更新控制器，因此下次迭代时，控制器就会给高准确率的结构更高的可能性。
也是用REINFORCE来训练， 
 - [DESIGNING NEURAL NETWORK ARCHITECTURES USING REINFORCEMENT LEARNING](https://arxiv.org/pdf/1611.02167v2.pdf)
来自MIT media lab．利用RL来优化CNN的结构。

 - [GENERATING LONG AND DIVERSE RESPONSES WITH NEURAL CONVERSATION MODELS]()
 
 - [Learning to compose words into sentences with reinforcement learning](https://arxiv.org/pdf/1611.09100v1.pdf)

这篇文章将强化学习用于句法分析
摘要：用强化学习学习树结构神经网络，用于计算自然语言句子的表示。之前关于树结构模型方面的工作，树被作为输入或者用来自显式treebank annotations的监督描述，本文的树结构被优化用于提升downstream task的性能。实验表明，学习task-specific composition顺序比基于treebank的序列编码和递归编码都要好。
有三种构建句子的向量表示的方法：1.RNN，将RNN最终的隐状态作为句子表示；2.tree-structured network递推的将词表示组成句子的表示，不同于序列模型，这种模型的结构根据句子的句法结构组织；3.用CNN以颠倒的方式构建表示。本文的工作可以看做前两个方法的折中，不用树结构显式的监督，而是用强化学习来学习树结构，将计算的句子表示作为reward signal。不同于序列RNN忽略树结构，我们的模型仍然为每个句子生成隐树，用它构建组合。我们的假设是
模型包括两部分：一个句子表示模型和一个用于学习树结构的强化学习算法，这个树结构在句子表示模型中使用。本文的句子表示模型遵循SPINN，SPINN是一个shift-reduce parser，采用LSTM作为它的组合函数。parser维护一个索引指针和一个栈，为了从句法上分析句子，parser会执行一系列操作，每个时间点的操作分SHIFT和REDUCE两种。SHIFT操作将词Xp推入栈，然后将指针移动到下一个词；REDUCE操作从栈中弹出两个元素，将它们组成一个元素压入栈。SHIFT操作在parse树中引入一个新的叶子节点，REDUCE操作将两个节点合并成一个成分。
强化学习：想法是用强化学习（policy gradient法）来发现最好的树结构，用Policy network来参数化action（SHIFT，REDUCE），也是采用REINFORCE算法来学习Wr，REINFORCE算法是policy gradient方法的一个例子。

 - [Framework of Automatic Text Summarization Using Reinforcement Learning](http://aclweb.org/anthology//D/D12/D12-1024.pdf)
 
 - [Human-level control through deep reinforcement learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)提出DQN模型
 - [On-line Active Reward Learning for Policy Optimisation in Spoken Dialogue Systems](https://arxiv.org/pdf/1605.07669v2.pdf)
 - [End-to-End Reinforcement Learning of Dialogue Agents for Information Access](https://arxiv.org/pdf/1609.00777v2)
 - [Gaussian processes for POMDP-based dialogue manager optimisation](http://mi.eng.cam.ac.uk/~sjy/papers/gayo14.pdf)
 - [Policy Learning for Domain Selection in an Extensible Multi-domain Spoken Dialogue System](https://www.baidu.com/link?url=ie-xHTHVQr-5UOaJ0WNMzP9EnRvSfnGvChbF9ON36jg6hfYal5vzRxogLhfOiSuwxE3ztrPV7YwbV5iA3H0lrK&wd=&eqid=ecc8c12600001d04000000035857ce37)
 - [End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning](https://arxiv.org/pdf/1606.01269.pdf)
 - [Towards End-to-End Learning for Dialog State Tracking and Management using Deep Reinforcement Learning](https://arxiv.org/pdf/1606.02560v2.pdf)
 - [Strategic Dialogue Management via Deep Reinforcement Learning](https://arxiv.org/pdf/1511.08099v1.pdf)
 - [Simultaneous Machine Translation using Deep Reinforcement Learning](http://tx.technion.ac.il/~danielm/icml_workshop/4.pdf)
 - [A Network-based End-to-End Trainable Task-oriented Dialogue System](https://arxiv.org/pdf/1604.04562v2.pdf)
 
