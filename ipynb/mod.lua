require 'nn'
require 'torch'

model = torch.load('opt_model')
print(model)