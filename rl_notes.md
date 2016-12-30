
## chapter 1

the idea that we learn by interacting with our environment is probably the first to occur to us when we think about the nature of 
learning. an infant plays.Learning from interaction is a foundational idea underlying nearly all theories of learning and intelligence.

In this book we explore a computational approach to learning from interaction

### 1.1 Reinforcement Learning
[ACM月刊强化学习的复兴](http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650718527&idx=1&sn=04db4fc59cc23c079a17573657d2b1c7&scene=21#wechat_redirect)

强化学习是一种学习模型，它并不会直接给你解决方案——你要通过“试错”去找到解决方案。深度强化学习与监督学习对比起来最清晰，监督学习是用来训练图像识别软件的，它以
有标注样本（标注需要人来打）的方式来监督。另一方面，强化学习不需要标签，也不需要通过输赢奖励来自动打标签。在强化学习中，你选择的行动越好，得到的反馈越多。
“不需要有人来告诉你什么好的行动什么是坏的行动，因为你能自己分辨它们——它能让你赢，它就是好的move”，但没那么简单，因为在每个动作和其回报之间有一个延迟，
这是强化学习的一个关键特征。“通常情况下，你必须经历上百个动作才能让得分上升”，如何区分哪些事让你的得分上升了，哪些事只是浪费时间呢？这个被称为`信用分配问题`，
仍然是强化学习的主要挑战，强化学习是唯一关注解决信用分配问题的机器学习领域，“如果是好的结果，我能回想起之前的n步，然后找出我走了哪些步，得到好的或者不好的结果吗？
深度学习受限于数据，如果使用强化学习自动生成数据，即使这些数据的标注比人类的标注弱很多，但因为我们自动生成它们，我们就可以得到更多的数据。

Jeff Dean:强化学习的想法是，你未必需要清楚你要采取的行动，所以你可以先做出一个你认为不错的行动，然后观察周围世界会有怎样的反应，这是一种探索行动序列的方式。
最后，在整个一系列的行动之后，你得到一些反馈信号，在你得到反馈信号的同时能将信用或责任分配到你所采取的所有行动。

 - 挑战：当你所处的行动状态非常宽泛时，此时使用强化学习会有一些挑战。

## survey
A dialogue policy is formulated as a Partially Observable Markov Decision Process(POMDP) which models the uncertainty existing in both the users' goals and the outputs of the ASR and the NLU.

http://www.leiphone.com/news/201612/RYp3OmyMycDSP5hd.html

http://www.leiphone.com/news/201611/TYsrMIlxkaROJ9q0.html

