#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.char_embed_size = 50
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, vocab.char2id['<pad>'])
        self.CNN = CNN(5, self.char_embed_size, self.embed_size)
        self.Highway = Highway(embed_size)
        self.dropout = nn.Dropout(0.3)

        # dropout layer

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        sentence_len = input.shape[0]
        batch_size = input.shape[1]

        input_reshaped = input.view(sentence_len * batch_size, input.shape[2])
        x_emb = self.embeddings(input_reshaped)
        x_reshaped = x_emb.permute(0, 2, 1)

        x_conv_out = self.CNN(x_reshaped)
        x_highway = self.Highway(x_conv_out)
        x_word_emb = self.dropout(x_highway)

        x_word_emb = x_word_emb.reshape(sentence_len, batch_size, x_word_emb.shape[1])
        return x_word_emb


        ### END YOUR CODE

