import os
import numpy as np
from src.inference.ewsnet import EWSNet

dataset = "W"
ensemble = 25
weight_dir = "./weights/Pretrained"
weight_dir = os.path.join(weight_dir,"Dataset-{}".format(dataset))

ewsnet = EWSNet(ensemble=ensemble, weight_dir=weight_dir)

x = np.random.randint(1,2,(20,10))
y = np.random.randint(0,3,(20,))
print(ewsnet.finetune(x,y))
