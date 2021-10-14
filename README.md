# TensorflowJS + Lobe.ai / Teachable Machine Demo

This is a fairly consice demo to show how you would add a model you have trained with [Teachable Machine](https://teachablemachine.withgoogle.com/) or [Lobe](https://lobe.ai/).

There isn't a huge amount of code, but the code itself is naturally quite specific to the [TensorflowJS API](https://js.tensorflow.org/api/latest/) so it might seem a bit weird. I've based this demo off the [MobileNet demo](https://github.com/tensorflow/tfjs-examples/tree/master/mobilenet) provided by the TensorflowJS team, and also the `tfjsExample.ts` script which you will find in the directory when you download the model from Lobe. But the latter just had a handy way of center cropping the input image, the __MobileNet__ demo is the important one. Why? The model you have trained using Teachable Machine or Lobe is a classification model, probably very similar to [MobileNet](https://arxiv.org/pdf/1704.04861.pdf). All we have done is restructure the last layer of the model to be more specific to the classes _you have defined_. 

So we can use a lot of that code to get us going, but we have to define the classes we used in training, which I have done at the top of `src/script.js`:

```javascript
const CLASSES = {
  0: 'hand',
  1: 'face',
}
```

There are some functions to handle the webcam input, some functions to run the model and obtain the results of the model and some others just for manipulating stuff in the DOM.

It's worth knowing that TensorflowJS (and every other machine learning framework for that matter) undergoes significant changes remarkably frequently; so there are often multiple ways of doing the same thing. _So_ a model can be export as either _Graph Model_ or a _Layers Model_ - which are just two ways of describing the inner structure of a model so that TensorflowJS can reconstruct it. __Lobe__ exports their models as a __Graph Model__ and so you would load the model like so:

```javascript
myModel = await tf.loadGraphModel(MODEL_PATH);
```

While __Teachable Machine__ exports their models as a __Layers Model__ which you would load like so:

```javascript
myModel = await tf.loadLayersModel(MODEL_PATH);
```

