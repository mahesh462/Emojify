# Emojify
Implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️).
- In many emoji interfaces, you need to remember that ❤️ is the "heart" symbol rather than the "love" symbol.
  - In other words, you'll have to remember to type "heart" to find the desired emoji, and typing "love" won't bring up that symbol.
- We can make a more flexible emoji interface by using word vectors!
- When using word vectors, you'll see that even if your training set explicitly relates only a few words to a particular emoji, the algorithm will be able to generalize and associate additional words in the test set to the same emoji.
  - This works even if those additional words don't even appear in the training set.
  - This allows you to build an accurate classifier mapping from sentences to emojis, even using a small training set.

1. First we'll start with a baseline model (Emojifier-V1) using word embeddings.
2. Then we will build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM.

# Baseline model: Emojifier-V1
##  Dataset EMOJISET
We have a tiny dataset (X, Y) where:
- X contains 127 sentences (strings).
- Y contains an integer label between 0 and 4 corresponding to an emoji for each sentence.
<p align = 'center'>
  <img src = '/images/data_set.png'>
</p>

## Overview of the Emojifier-V1
<p align = 'center'>
  <img src = '/images/emojifierv1.png'>
</p>

### Inputs and outputs
- The input of the model is a string corresponding to a sentence (e.g. "I love you).
- The output will be a probability vector of shape (1,5), (there are 5 emojis to choose from).
- The (1,5) probability vector is passed to an argmax layer, which extracts the index of the emoji with the highest probability.

To get our labels into a format suitable for training a softmax classifier, convert Y from its current shape (m,1) into a "one-hot representation"  (m,5). Each row is a one-hot vector giving the label of one example.

As shown in above figure, the first step is to:
1. Convert each word in the input sentence into their word vector representations.
2. Then take an average of the word vectors.
3. We will use pre-trained 50-dimensional GloVe embeddings.

## Model
After averaging word vectors we need to:
1. Pass the average through forward propagation.
2. Compute the cost.
3. Backpropagate to update the parameters.

After implementing the above model we got an accuracy of:
- Training set accuracy: 97.79%
- Test set accuracy    : 85.7%

## Word Orderings
The above model don't follow the word ordering. It understands "not feeling happy" as "feeling happy" and returns same emoji for both the sentences.

So, the above algorithm is not good for phrases like "not feeling happy".

# Emojifier-V2: Using LSTMs in Keras:
LSTM model that takes word sequences as input!
- This model will be able to account for the word ordering.
- Emojifier-V2 will continue to use pre-trained word embeddings to represent words.
- We will feed word embeddings into an LSTM.
- The LSTM will learn to predict the most appropriate emoji.

## Overview of the model
<p align = 'center'>
  <img src = '/images/emojifier-v2.png'>
</p>

## Keras and mini batching

- We want to train Keras using mini-batches.
- However, most deep learning frameworks require that all sequences in the same mini-batch have the same length.
  - This is what allows vectorization to work: If you had a 3-word sentence and a 4-word sentence, then the computations needed for them are different (one takes 3 steps of an LSTM, one takes 4 steps) so it's just not possible to do them both at the same time.

### Padding handles sequences of varying length
The common solution to handling sequences of different length is to use padding. Specifically:
1. Set a maximum sequence length.
2. Pad all sequences to have the same length.

### Example of Padding:
- Given a maximum sequence length of 20, we could pad every sentence with "0"s so that each input sentence is of length 20.
- Thus, the sentence "I love you" would be represented as  (e<sub>I</sub>,e<sub>hate</sub>,e<sub>you</sub>,<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{0}{\rightarrow}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{0}{\rightarrow}" title="\underset{0}{\rightarrow}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\underset{0}{\rightarrow}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{0}{\rightarrow}" title="\underset{0}{\rightarrow}" /></a> ,…,<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{0}{\rightarrow}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{0}{\rightarrow}" title="\underset{0}{\rightarrow}" /></a>).


## The Embedding layer
- In Keras, the embedding matrix is represented as a "layer".
- The embedding matrix maps word indices to embedding vectors.
  - The word indices are positive integers.
  - The embedding vectors are dense vectors of fixed size.
  - When we say a vector is "dense", in this context, it means that most of the values are non-zero. As a counter-example, a one-hot encoded vector is not "dense."
- The embedding matrix can be derived in two ways:
  1. Training a model to derive the embeddings from scratch.
  2. Using a pretrained embedding.

We build the Embedding() layer in Keras, using pre-trained word vectors. The embedding layer takes as input a list of word indices and return the word embeddings for a sentence


## Building the Emojifier-V2
Feed the embedding layer's output to an LSTM network.
<p align = 'center'>
  <img src = '/images/emojifier-v2.png'>
</p>

After creating the model in Keras, we need to compile it and define what loss, optimizer and metrics your are want to use.
1. Loss      = "categorical_crossentropy"
2. Optimizer = "adam"

After training the above Emojifier-v2 model, accuracy will be:

- Training set accuracy: around 90% to 100%
- Test set accurcy     : 83.92% (between 80% to 95%)

1. Previously, Emojify-V1 model did not correctly label "not feeling happy," but our implementation of Emojiy-V2 got it right.
2. The current model still isn't very robust at understanding negation (such as "not happy").
  - This is because the training set is small and doesn't have a lot of examples of negation.
  - But if the training set were larger, the LSTM model would be much better than the Emojify-V1 model at understanding such complex sentences.





