# Cat_or_dog-kaggle-vgg16-tensorflow
A simple Tensorflow code for fine-tune VGG-16 to solve 'cat or dog' task in kaggle.

## Requirements

Tensorflow > 1.3

Python > 2.7

vgg_16.ckpt from Tensorflow slim pre-trained model https://github.com/tensorflow/models/tree/master/research/slim

Data sets from kaggle https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

You should place vgg_16.ckpt in 'check_point' file folder and data in the 'data' file folder

## Details

Fine-tune the last layer of VGG16 to tell cat or dog images. I extract 12500(30% of train set) validation samples from train set. 

Batch size: 32

learning rate: 1e-4

dropout: 0.5

No other settings in order to make a simple initialization.

## Run

Train: python train.py

Test and obtain result: python test.py

## output

97% accuracy on validation set.
