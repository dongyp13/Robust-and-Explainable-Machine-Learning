# Robust-and-Explainable-machine-learning
Related materials for robust and explainable machine learning

## Contents 

- [Robustness](#robustness)
- [Explainability](#explainability)

## Robustness
### Properties
* [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) <br/> Individual unit contains no semantic information; Adversarial examples by L-BFGS (Optimization based).
* [Deep Neural Networks are Easily Fooled:
High Confidence Predictions for Unrecognizable Images](https://arxiv.org/abs/1412.1897) <br/> Fool images by evolution algorithm.
* [Universal adversarial perturbations](https://arxiv.org/abs/1610.08401) <br/> Universal adversarial perturbations can fool the network in most of the images.

### Transferability
* [Delving into Transferable Adversarial Examples and Black-box Attacks](https://arxiv.org/abs/1611.02770) <br/> Examine the transferability on ImageNet dataset and use this property to attack black-box systems.


### Attack
* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) <br/> Fast gradient sign method.
* [Adversarial Examples In The Physical World](https://arxiv.org/abs/1607.02533) <br/> Printed photos can also fool the networks; Introduce an iterative method (extension of FGS).
* [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528) <br/> Find salient input regions that are useful for adversarial examples.
* [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) <br/> Optimization based approach.
* [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/pdf/1511.04599.pdf) <br/> A new method to generate non-targeted adversarial examples. Find the closest boundary and also use the gradient.
* [Good Word Attacks on Statistical Spam Filters](http://www.egov.ufsc.br/portal/sites/default/files/anexos/5867-5859-1-PB.pdf)
* [Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples](https://arxiv.org/abs/1602.02697) <br/> Block-box attack using a substitute network.
* [Simple Black-Box Adversarial Perturbations for Deep Networks](https://arxiv.org/abs/1612.06299) <br/> Black-box attack using greedy search.
* [Adversarial Manipulation of Deep Representations](https://arxiv.org/abs/1511.05122) <br/> Find an adversarial image that has similar representations with a target image (trivial).
* [Adversarial Diversity and Hard Positive Generation](https://arxiv.org/abs/1605.01775)


### Generative Model
* [Adversarial examples for generative models](https://arxiv.org/abs/1702.06832) <br/> Attack VAE and VAE-GAN.
* [Adversarial Images for Variational Autoencoders](https://arxiv.org/abs/1612.00155) <br/> Attack VAE by latent representations. 

### Defense
* [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://arxiv.org/abs/1511.04508) <br/> Train a second network with soft target labels.
* [Robust Convolutional Neural Networks under Adversarial Noise](https://arxiv.org/abs/1511.06306) <br/> Improve robustness by injecting noise during training.
* [Towards Deep Neural Network Architectures Robust to Adversarial Examples](https://arxiv.org/abs/1412.5068) <br/> Use aotoencoder to denoise.
* [On Detecting Adversarial Perturbations](https://arxiv.org/abs/1702.04267) <br/> Detect adversarial perturbations in intermediate layers by a detector network and dynamic generate adversarial images during training. They also propose fast gradient method, which is an extension of iterative method based on l2 norm.

### Theoretical Attack
* [Measuring Neural Net Robustness with Constraints](https://arxiv.org/pdf/1605.07262.pdf)<br/>A measurement of robustness.
* [A Theoretical Framework for Robustness of (Deep) Classifiers against Adversarial Examples](https://arxiv.org/abs/1612.00334)
* [Blind Attacks on Machine Learners](https://papers.nips.cc/paper/6482-blind-attacks-on-machine-learners)
* [SoK Towards the Science of Security and Privacy in Machine Learning](https://spqr.eecs.umich.edu/papers/rushanan-sok-oakland14.pdf)
* [Robustness of classifiers: from adversarial to random noise](https://arxiv.org/abs/1608.08967)

## Explainability
### Visualization
* [Visualizing and Understanding Convolutional Networks
] (https://arxiv.org/abs/1311.2901) <br/> Code inversion.
* [Inverting Visual Representations with Convolutional Networks](https://arxiv.org/abs/1506.02753) <br/> Code inversion by learning a decoder network.
* [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035) <br/> Code inversion with priors.
* [Synthesizing the preferred inputs for neurons in neural networks via deep generator networks](https://arxiv.org/abs/1605.09304) <br/> Synthesize an image from internal representations and use GAN (deconvolution) to learn image priors. (like code inversion)
* [Visualizing Higher-Layer Features of a Deep Network](https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network) <br/> Activation maximization.
* [Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks](https://arxiv.org/pdf/1602.03616.pdf) <br/> Activation maximization for multifaceted features.
* [Towards Better Analysis of Deep Convolutional Neural Networks](https://arxiv.org/abs/1604.07043) <br/> An useful tool. Represent a neuron by top image patches with highest activation.
* [Object Detectors Emerge in Deep Scene CNNs](https://arxiv.org/abs/1412.6856) <br/> Visualize neurons by highest activated images and corresponding receptive fields.
* [Visualizing Deep Neural Network Decisions: Prediction Difference Analysis](https://arxiv.org/abs/1702.04595) <br/> A general method to visualize image regions that support or against a prediction (Attention). It can also be used to visualize neurons.

### Attention
* [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) <br/> CAM for weakly supervised detection.
* [Grad-CAM: Why did you say that?
Visual Explanations from Deep Networks via Gradient-based Localization] (https://arxiv.org/abs/1610.02391) <br/> Extension of CAM on captioning and VQA.

### Justification
* [Generating Visual Explanations](https://arxiv.org/abs/1603.08507) <br/> Generate an explanation for bird classification.
* [Attentive Explanations: Justifying Decisions and Pointing to the Evidence](https://arxiv.org/abs/1612.04757) <br/> Justify its decisions by generating a neuron sentence and pointing to important image regions (Attention) in VQA task. 

### Generative Models
* [Inducing Interpretable Representations with Variational Autoencoders](https://arxiv.org/abs/1611.07492) <br/> Learn interpretable latent variables in VAE.
* [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) <br/> In GAN.
