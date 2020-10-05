# TactileSGNet 
This repository contains code for the IROS 2020 paper "TactileSGNet: A Spiking Graph Neural Network for Event-based Tactile Object Recognition". In this paper, we propose a novel spiking graph neural network for event-based tactile object recognition. To make use of local connectivity of taxels, we present several methods for organizing the tactile data in a graph structure. Based on the constructed graphs, we develop a spiking graph convolutional network. The event-driven nature of spiking neural network makes it arguably more suitable for processing the event-based data. Experimental results on two tactile datasets show that the proposed method outperforms other state-of-the-art spiking methods, achieving high accuracies of approximately 90% when classifying a variety of different household objects.

# Dependencies
- Pytorch (tested on v1.4.0) 
- torchvision 
- Numpy 
- torch_geometric
- tqdm
- scikit-learn
- CUDA 10
- time

# Usage
- Download Ev-Touch Objects dataset, and put it under the same folder as these python files
- Run the following 'main.py' to see the result

# Questions?
For any questions regarding the code or the paper, please email me at gufq87 at gmail.com.

# Citing
```
@inproceedings{gu2020tactilesgnet,
  title={TactileSGNet: A Spiking Graph Neural Network for Event-based Tactile Object Recognition},
  author={Gu, Fuqiang and Sng, Weicong and Taunyazov, Tasbolat and Soh, Harold},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```
