#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

class CNN(nn.Module):
	def __init__(self, kernel_size, char_embed_size, word_embed_size):
		# kernel should be 5
		# num_filters is the word embed size
		super(CNN, self).__init__()
		self.conv = nn.Conv1d(char_embed_size, word_embed_size, kernel_size)
		self.maxPool = nn.MaxPool1d(21 - kernel_size + 1)

	def forward(self, x_reshaped):
		x_conv = self.conv(x_reshaped)
		x_conv_out = self.maxPool(F.relu(x_conv))
		x_conv_out = torch.squeeze(x_conv_out, 2)
		return x_conv_out

### END YOUR CODE

