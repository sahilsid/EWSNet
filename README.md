# EWSNet 

[![DOI](https://zenodo.org/badge/338530625.svg)](https://zenodo.org/badge/latestdoi/338530625)

### Setup Instructions
Create a new conda environment with tensorflow using 
  - If you have a CPU machine
```shell
conda create -n ENV_NAME tensorflow
```
- If you have a gpu machine

```shell
conda create -n ENV_NAME tensorflow-gpu
```

Activate the environment and install  the required packages
```shell
conda activate ENV_NAME
pip install -r requirements.txt 
	   
```

Alternatively use can also setup a conda environment with all the required packages using the *environment.yml* file.
```shell

conda env create -f environment.yml
	   
```

### Testing for transitions with EWSNet
Download model weights from  [ewsnet-weights](https://drive.google.com/file/d/1-aY2MepouLQdMSNkYD6jgSedwFXB8BUP/view?usp=sharing "ewsnet-weights") and extract it inside the weights folder.

To test the ewsnet model on custom time series data, create an instance of EWSNet class as provided in ewsnet.py, passing the weight directory (chose any of Dataset-C or Dataset-W ) and call the predict function on the input. Here is a sample code 


```python
import os
import numpy as np
from src.inference.ewsnet import EWSNet

dataset = "W"
ensemble = 25
weight_dir = "./weights/Pretrained"
weight_dir = os.path.join(weight_dir,"Dataset-{}".format(dataset))

ewsnet = EWSNet(ensemble=ensemble, weight_dir=weight_dir)

x = np.random.randint(1,2,(20,))
print(ewsnet.predict(x))
```

The arguments denote :
- ***ensemble*** : The no. of trained models to average the prediction over. (between 1 and 25)
- ***weight_dir*** : The directory which contains the weights for Dataset-C and Dataset-W 
- ***prefix***         : The prefix for individual weight filenames. Defaults to empty prefix
- ***suffix***         : The suffix for individual weight filenames. Defaults ".h5"

----
### Finetuning EWSNet

To finetune the ewsnet model on custom time series data, create an instance of EWSNet class as provided in ewsnet.py, passing the weight directory (chose any of Dataset-C or Dataset-W ) and call the finetune() function, by providing the dataset and other training parameters. Here is a sample code 


```python
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

```
The arguments denote :
- ***X*** : The data points (univariate timeseries) to finetune EWSNet on. Dimension - (N x D) or (N x 1 x D) where `N` denotes the no. of samples and `D` denotes the no. of time steps.

- ***y*** : The target labels corresponding to the data points (X). Dimension - (N, ) or (N x 1) where `N` denotes the no. of samples.

- ***freeze_feature_extractor*** : A boolean flag that determines the part of the network to be finetuned. When set to False. the entire network is finetuned. When set to True, only the fully connected layers are finetuned and the feature extraction blocks are frozen.

- ***learning_rate*** : The learning rate for finetuning the models.

- ***batch_size*** : The batch size for finetuning the models.

- ***tune_epochs*** : The no. of epochs for finetuning the models.

----

### Real-World Data

A few Real-world paleoclimatic and ecological data are added in the *data* folder for testing. You may use the code snippet below to load them and predict using ewsnet. 

```python
import numpy as np
import pandas as pd 
from src.inference.ewsnet import EWSNet
import os

weight_dir = "./weights/Pretrained"
dataset    = "W"
prefix     = ""
suffix     = ".h5"
ensemble   = 25

ewsnet     = EWSNet(ensemble=ensemble, weight_dir=os.path.join(weight_dir,"Dataset-{}".format(dataset)), prefix=prefix,suffix=suffix)

data_dir = "./data"
for filename in os.listdir(data_dir):
    series = np.loadtxt(os.path.join(data_dir,filename))
    name = filename.split(".")[0]
    print("\n\n==>Testing Data : ",name)
    label,prob = ewsnet.predict(series)
    print("----- Predicted Label : ",label)
    print("----- Prediction Probability : \n \t ",prob,"\n")
```
