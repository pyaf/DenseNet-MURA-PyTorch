# DenseNet on MURA Dataset using PyTorch

A PyTorch implementation of 169 layer [DenseNet](https://arxiv.org/abs/1608.06993) model on MURA dataset, inspired from the paper [arXiv:1712.06957v3](https://arxiv.org/abs/1712.06957) by Pranav Rajpurkar et al. MURA is a large dataset of musculoskeletal radiographs, where each study is manually labeled by radiologists as either normal or abnormal. [know more](https://stanfordmlgroup.github.io/projects/mura/)

## Important Points:
* The implemented model is a 169 layer DenseNet with single node output layer initialized with weights from a model pretrained on ImageNet dataset.
* Before feeding the images to the network, each image is normalized to have same mean and standard deviation as of the images in the ImageNet training set, scaled to 224 x 224 and augmentented with random lateral inversions and rotations.
* The model uses modified Binary Cross Entropy Loss function as mentioned in the paper.
* The Learning Rate decays by a factor of 10 every time the validation loss plateaus after an epoch.
* The optimization algorithm is Adam with default parameters β1 = 0.9 and β2 = 0.999.

According to MURA dataset paper:

> The model takes as input one or more views for a study of an upper extremity. On each view, our 169-layer convolutional neural network predicts the probability of abnormality. We compute the overall probability of abnormality for the study by taking the arithmetic mean of the abnormality probabilities output by the network for each image.

The model implemented in [model.py](model.py) takes as input 'all' the views for a study of an upper extremity. On each view the model predicts the probability of abnormality. The Model computes the overall probability of abnormality for the study by taking the arithmetic mean of the abnormality probabilites output by the network for each image.

## Instructions

Install dependencies:
* PyTorch
* TorchVision
* Numpy
* Pandas

Train the model with `python main.py`

## Citation
    @ARTICLE{2017arXiv171206957R,
       author = {{Rajpurkar}, P. and {Irvin}, J. and {Bagul}, A. and {Ding}, D. and 
      {Duan}, T. and {Mehta}, H. and {Yang}, B. and {Zhu}, K. and 
      {Laird}, D. and {Ball}, R.~L. and {Langlotz}, C. and {Shpanskaya}, K. and 
      {Lungren}, M.~P. and {Ng}, A.},
        title = "{MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs}",
      journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
       eprint = {1712.06957},
     primaryClass = "physics.med-ph",
     keywords = {Physics - Medical Physics, Computer Science - Artificial Intelligence},
         year = 2017,
        month = dec,
       adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171206957R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
