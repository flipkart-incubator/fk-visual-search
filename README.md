# fk-visual-search
This code allows you to train the Visnet model. VisNet is a Convolutional Neural Network (CNN) trained using triplet based deep ranking paradigm. It contains a deep CNN modelled after the VGG-16 network, coupled with parallel shallow convolution layers in order to capture both high-level and low-level image details simultaneously. Visnet, trained on Flipkart's proprietary internal dataset, powers Visual Recommendations at Flipkart. On the publically available dataset, [Street2Shop](http://tamaraberg.com/street2shop/), Visnet achieves state-of-the-art results.

In this Repo, we have open-sourced the training prototxts of Visnet. We soon plan to add other useful scripts, such as:
* Triplet sampling code, to generate the training files
* A CUDA based fast K-Nearest Neighbor Search library
* Other auxillary scripts, such as code to process Street2Shop dataset, compute recall, etc
* Our useful modifications over Caffe, such as the image augmentation layer, and triplet accuracy layer to aid the training of Visnet
