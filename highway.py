#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class Highway(nn.Module):
	def __init__(self, embed_size):
		# Use two linear layers
		super(Highway, self).__init__()
		self.proj = torch.nn.Linear(embed_size, embed_size)
		self.gate = torch.nn.Linear(embed_size, embed_size)

	def forward(self, x_conv):
		# Maps from x_conv_out to x_highway

		# apply RELU on layer 1
		# apply sigmoid on layer 2
		# element wise multiply them

		x_proj = F.relu(self.proj(x_conv))
		x_gate = torch.sigmoid(self.gate(x_conv))

		x_highway = x_gate * x_proj + (1 - x_gate) * x_conv
		return x_highway




### END YOUR CODE 

