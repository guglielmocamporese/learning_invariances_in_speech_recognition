# Learning Invariances In Speech Recognition

### Abstract
In this work I investigate the speech command task developing and analyzing deep learning models. The state of the art technology uses convolutional neural networks (CNN) because of their intrinsic nature of learning correlated represen- tations as is the speech. In particular I develop different CNNs trained on the Google Speech Command Dataset and tested on different scenarios. A main problem on speech recognition consists in the differences on pronunciations of words among different people: one way of building an invariant model to variability is to augment the dataset perturbing the input. In this work I study two kind of augmentations: the Vocal Tract Length Perturbation (VTLP) and the Synchronous Overlap and Add (SOLA) that locally perturb the input in frequency and time respectively. The models trained on augmented data outperforms in accuracy, precision and recall all the models trained on the normal dataset. Also the design of CNNs has impact on learning invariances: the inception CNN architecture in fact helps on learning features that are invariant to speech variability using different kind of kernel sizes for convolution. Intuitively this is because of the implicit capability of the model on detecting different speech pattern lengths in the audio feature.

# Data Augmentation
### Data Augmentation Scheme
![data_aug](https://user-images.githubusercontent.com/31989563/42893409-85d7c284-8ab4-11e8-9e01-51fe66e9e629.png)

### Data Augmentation Example
![aug_example](https://user-images.githubusercontent.com/31989563/42893425-902d0d70-8ab4-11e8-9329-ddd67988be6a.png)

# Deep Learning Models Used
### Convolutional Neural Network Model
![cnn](https://user-images.githubusercontent.com/31989563/42893350-65d2355a-8ab4-11e8-94da-98d53cd41f9a.png)

### Feedforward Neural Network Autoencoder Model
![fnn_ae](https://user-images.githubusercontent.com/31989563/42893379-777b54c6-8ab4-11e8-85cc-c22b4d5d9d48.png)

### Convolutional Neural Network Autoencoder Model
![cnn_ae](https://user-images.githubusercontent.com/31989563/42893137-d8e35840-8ab3-11e8-961a-96cf734cc56c.png)

### Convolutional Neural Network Inception Model
![cnn_inc](https://user-images.githubusercontent.com/31989563/42893367-6fbb8f76-8ab4-11e8-8872-1650ec0facbf.png)

# Results
![results](https://user-images.githubusercontent.com/31989563/42893680-53b8f01a-8ab5-11e8-8370-d70820df5555.png)
