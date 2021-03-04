# EWSNet

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
Download model weights from  [ewsnet-weights](https://drive.google.com/file/d/1-aY2MepouLQdMSNkYD6jgSedwFXB8BUP/view?usp=sharing "ewsnet-weights") and extract it inside the weights folder.

To test the ewsnet model on custom time series data, create an instance of EWSNet class as provided in ewsnet.py, passing the weight directory (chose any of Dataset-C or Dataset-W ) and call the predict function on the input. Here is a sample code 


```python
import os
import numpy as np
from ewsnet import EWSNet

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
- ***prefix***         : The prefix for individual weight filenames. Defaults ""
- ***suffix***         : The suffix for individual weight filenames. Defaults ".h5"

The utilities of other files are summrized below: 
- **main.ipynb** : Use to run experiments for training the EWSNet models
- **generic_ews.ipynb** : Use to run experiments for training the Classical ML models

