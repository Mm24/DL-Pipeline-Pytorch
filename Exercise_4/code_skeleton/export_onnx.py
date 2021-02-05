import torch as t
from Exercise_4.code_skeleton.trainer import Trainer
from Exercise_4.code_skeleton.model import ResNet
import sys


epoch = 40
#TODO: Enter your model here

model = ResNet()
crit = t.nn.BCELoss()


trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('saved_model/checkpoint_{:03d}.onnx'.format(epoch))
