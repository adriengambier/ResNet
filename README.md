# ResNet

This project is the implementation of the [ResNet paper](https://arxiv.org/pdf/1512.03385.pdf) for CIFAR10. It is meant to train the model from scratch with the CIFAR10 dataset and reproduce results from the paper.

All the code is done with Keras and different architectures can be tested as presented in the paper (networks with 20, 32, 44, 56 and 110 layers).

# Performance

ResNet with 32 layers showed similar results with this implementation, the accuracy during training is shown below :

![Alt text](images/accuracy_training.PNG?raw=true "Title")

The model was trained once and closer results could be obtained after multiple training. <br>
Note that this implementation uses projection shortcuts to increase dimensions, meaning 1x1 convolution on the shortcut connection. 

| # Layers | Paper (%error)  | This implementation (%error)  |
| -------- | --------------- | ----------------------------- |
| 32       | 7.51            | 7.78                          |

# Usage

An instance of ResNet class must be created to train a model.

```
from resnet_keras import ResNetCIFAR

resnet = ResNetCIFAR()
# N is the parameter for the number of layers (#layers = 6N + 2)
resnet.create_model(N=5)
```

To train it, a train_set and valid_set generator are used as inputs :

```
resnet.train_model(train_batches, valid_batches, num_iterations=60000)
```

Finally, a model can be saved and the training can be resumed at where it was stopped : 

```
resnet.save_model('my_model.zip')

resnet_load = ResNetCIFAR.load_model('my_model.zip')

resnet_load.train_model(train_batches, valid_batches, num_iterations=60000, init_iter=15000)
```
