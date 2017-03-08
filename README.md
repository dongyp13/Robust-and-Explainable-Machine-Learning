# Robust-and-Explainable-machine-learning
Related materials for robust and explainable machine learning

## Contents 

- [Robustness](#robustness)
- [Explainability](#explainability)


##Explainability
###Visualization
* [Visualizing and Understanding Convolutional Networks] (https://arxiv.org/abs/1311.2901) <br/> Code inversion.
* [Inverting Visual Representations with Convolutional Networks] (https://arxiv.org/abs/1506.02753) <br/> Code inversion by learning a decoder network.
* [Understanding Deep Image Representations by Inverting Them] (https://arxiv.org/abs/1412.0035) <br/> Code inversion with priors.
* [Synthesizing the preferred inputs for neurons in neural networks via deep generator networks](https://arxiv.org/abs/1605.09304) <br/> Synthesize an image from internal representations and use GAN (deconvolution) to learn image priors. (like code inversion)
* [Visualizing Higher-Layer Features of a Deep Network](https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network) <br/> Activation maximization.
* [Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks] (https://arxiv.org/pdf/1602.03616.pdf) <br/> Activation maximization for multifaceted features.
* [Towards Better Analysis of Deep Convolutional Neural Networks] (https://arxiv.org/abs/1604.07043) <br/> An useful tool. Represent a neuron by top image patches with highest activation.
* [Object Detectors Emerge in Deep Scene CNNs] (https://arxiv.org/abs/1412.6856) <br/> Visualize neurons by highest activated images and corresponding receptive fields.
* [Visualizing Deep Neural Network Decisions: Prediction Difference Analysis] (https://arxiv.org/abs/1702.04595) <br/> A general method to visualize image regions that support or against a prediction (Attention). It can also be used to visualize neurons.

###Attention
* [Learning Deep Features for Discriminative Localization] (http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) <br/> CAM for weakly supervised detection.
* [Grad-CAM: Why did you say that?Visual Explanations from Deep Networks via Gradient-based Localization] (https://arxiv.org/abs/1610.02391) <br/> Extension of CAM on captioning and VQA.

###Justification
* [Generating Visual Explanations](https://arxiv.org/abs/1603.08507) <br/> Generate an explanation for bird classification.
* [Attentive Explanations: Justifying Decisions and Pointing to the Evidence](https://arxiv.org/abs/1612.04757) <br/> Justify its decisions by generating a neuron sentence and pointing to important image regions (Attention) in VQA task. 

###Generative Models
* [Inducing Interpretable Representations with Variational Autoencoders](https://arxiv.org/abs/1611.07492) <br/> Learn interpretable latent variables in VAE.
* [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets] (https://arxiv.org/abs/1606.03657) <br/> In GAN.