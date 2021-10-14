import 'regenerator-runtime/runtime' // Just to prevent a error from Parcel
import * as tf from '@tensorflow/tfjs';

const MODEL_PATH = './hand_face_tm/model.json';
const VIDEO_ELEM = document.getElementsByTagName('video')[0]
const CANVAS = document.getElementsByTagName('canvas')[0]
const BTN_ELEM = document.getElementsByTagName('button')[0]
const EMOJI_ELEM = document.getElementById('emoji')
const CTX = CANVAS.getContext('2d')
const IMAGE_SIZE = 224
const TOPK_PREDICTIONS = 2

const CLASSES = {
  0: 'hand',
  1: 'face',
}

let handFaceModel;

function initCam(videoElem) {
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then(function (stream) {
        videoElem.srcObject = stream;
      })
      .catch(function (error) {
        console.log('Something went wrong with the webcam...');
        console.log(error);
      });
  }
}

function stopCam(videoElem) {
  const stream = videoElem.srcObject;
  const tracks = stream.getTracks();

  for (let i = 0; i < tracks.length; i++) {
    tracks[i].stop();
  }

  videoElem.srcObject = null;
}

function isVideoPlaying(video) {
  return !!(video.currentTime > 0 && !video.paused && !video.ended && video.readyState > 2)
}

function status(text) {
  const statusElem = document.getElementById('status')
  statusElem.innerText = text
}

async function loadModel() {
  status('Loading model...');

  handFaceModel = await tf.loadLayersModel(MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  handFaceModel.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');
  BTN_ELEM.classList.remove('hide')
  BTN_ELEM.classList.add('button')
};

async function predict(imageTensor){
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    //const img = tf.cast(tf.browser.fromPixels(imgElement), 'float32');

    const [imgHeight, imgWidth] = imageTensor.shape.slice(0,2); // convert image to 0-1
    const normalizedImage = tf.div(imageTensor, tf.scalar(255));
    const reshapedImage = normalizedImage.reshape([1, ...normalizedImage.shape]);
    let top = 0;
    let left = 0;
    let bottom = 1;
    let right = 1;
    if (imgHeight != imgWidth) {
      // the crops are normalized 0-1 percentage of the image dimension
      const size = Math.min(imgHeight, imgWidth);
      left = (imgWidth - size) / 2 / imgWidth;
      top = (imgHeight - size) / 2 / imgHeight;
      right = (imgWidth + size) / 2 / imgWidth;
      bottom = (imgHeight + size) / 2 / imgHeight;
    }
    const croppedImage = tf.image.cropAndResize(
      reshapedImage, [[top, left, bottom, right]], [0], [IMAGE_SIZE, IMAGE_SIZE]
    );


    startTime2 = performance.now();
    // Make a prediction 
    return handFaceModel.predict(croppedImage);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` + `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  return classes
}

async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

function process(result) {
  const [a, b] = result
  if(a.probability > b.probability) {
    if(a.className === 'hand'){
      // Hand
      console.log('Hand')
      EMOJI_ELEM.innerText = String.fromCodePoint(0x1F590)
      EMOJI_ELEM.classList.toggle('left', true)
      EMOJI_ELEM.classList.toggle('right', false)
    } else {
      // Face
      console.log('Face')
      EMOJI_ELEM.innerText = String.fromCodePoint(0x1F468)
      EMOJI_ELEM.classList.toggle('left', false)
      EMOJI_ELEM.classList.toggle('right', true)
    }
  }
}

async function main() {
  initCam(VIDEO_ELEM)
  await loadModel()
  BTN_ELEM.onclick = run
}

async function run() {
  const decodedImage = await tf.browser.fromPixelsAsync(VIDEO_ELEM, 3);
  const result = await predict(decodedImage)
  process(result)
}

main()
