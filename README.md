## DDI-MDAE and DDI-MDAE (PU learning)

This repository proposed a multi-modal deep auto-encoders based drug representation learning method for DDI prediction, named DDI-MDAE. Moreover, we improved DDI-MDAE by applying a specialized random forest classifier in the positive-unlabeled (PU) learning setting, the improved method is called DDI-MDAE (PU learning). 

### Input format

The input data should be an undirected graph in which node IDs start from 1 to N (N is the number of nodes in the graph). Each line contains two node IDs of drug and feature repectively indicating an edge in the graph.

Text file sample

```0 1```  
```2 5```  
```...```

### Environments
- Python 3.6.8 :: Anaconda, Inc.
- Tensorflow 1.14.0

### Usage

When using this code, you need to clone this repo and load all the files in the folder into your running environment first. Then, you need to adjust the pu parameter in the following line of code in `pu_main.py`.   
```CV_num, method_num, alpha, beta, gama, mu, dimension, drug_size, removed_ratio, learning_rate, pu = [3, 4, 0.1, 2, 0.1, 1e-5, 128, 2367, 0, 0.001, 1]```  

Next, you need enter the root directory and run the following code:  
```cd src```  
``` python pu_main.py``` 

When the parameter pu is 0, the file `pu_main.py` implements the DDI-MDAE method to predict potential interactions between drug-drug pairs, and when the parameter pu is 1, the file `pu_main.py` implements DDI-MDAE (PU-learning) method to predict potential interactions between drug-drug pairs.
