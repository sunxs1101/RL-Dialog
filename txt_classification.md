## use CNN for text classification

### note
They train a simple CNN with one layer of convolution on top of word vectors obtained from an unsupervised neural language model for sentence-level classification tasks. Word vectors, wherein words are projected from a sparse, 1-of-V encoding(V is the vocabulary size) onto a lower dimensinal vector space via a hidden layer, are essentially feature extractors that encode semantic features of words in their dimensions. In such dense representations, semantically close words are likewise close-in euclidean or cosine distance-in the lower dimensional vector space. These vectors were trained by Mikolov on 100 billion words of Google News, and are publicly available. They use these publicly available word2vec vectors.


Instead of image pixels, the input to most NLP tasks are sentences or documents represented as a matrix. Each row of the matrix corresponds to one token, typically a word, but it could be a character. That is, each row is vector that represents a word. Typically, these vectors are word embeddings (low-dimensional representations) like word2vec or GloVe, but they could also be one-hot vectors that index the word into a vocabulary. For a 10 word sentence using a 100-dimensional embedding we would have a 10×100 matrix as our input. That’s our “image”.

### reference
 - [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
 - [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
 - [A Neural Probabilistic Language Model](http://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
 - [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)

#### RNN
[recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)

