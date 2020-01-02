# RetinaNet for Cell Detection

Keras implementation of RetinaNet for cell detection adapted from [Fizyr-RetinaNet](https://github.com/fizyr/keras-retinanet).

Reference paper: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## Notes
This repository was tested using:
* Python 3.6.8
* Keras 2.2.5
* Tensorflow-gpu 1.14
* OpenCV 4.1.1.26

## Data augmentation
We applied some techniques for data augmentation, including the PSF kernels to simulate image motion:
* Gaussian
* Line and curve
* Airy disk pattern
* Smooth elastic transform

 <div class="row">
  <div class="column">
    <img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_3a.png" alt="Gaussian kernel example." style="width:100%">
  </div>
  <div class="column">
    <img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_3b.png" alt="Line kernel example." style="width:100%">
  </div>
  <div class="column">
    <img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_3c.png" alt="Curve kernel example." style="width:100%">
  </div>
  <div class="column">
    <img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_4b.png" alt="Airy disk pattern." style="width:100%">
  </div>
</div> 

<div class="row">
  <div class="column">
    <img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_5a.png" alt="Example of an original image." style="width:100%">
  </div>
  <div class="column">
    <img src="https://github.com/brunoggregorio/retinanet-cell-detection/blob/master/images/Fig_5b.png" alt="Image after deformable elastic transform." style="width:100%">
  </div>

#### Backbones
 Visualizations of network structures (tools from [ethereon](http://ethereon.github.io/netscope/quickstart.html)):

    - [ResNet-50] (http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)
	- [ResNet-101] (http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50)
	- [ResNet-152] (http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b)


Ref: https://github.com/KaimingHe/deep-residual-networks
