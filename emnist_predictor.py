from tensorflow.contrib import predictor
import imageio
import tensorflow as tf
from PIL import Image
import numpy as np

img = imageio.imread('seven.png')
img = np.asarray(img, dtype=np.float32)
img /= 255

img = img[:,:,0]
img /= 255
predict_fn = predictor.from_saved_model('./tmp/saved_model/1523348492',signature_def_key='predict')
def serving_input_rc_fn():
  """Build the serving inputs."""
  # The outer dimension (None) allows us to batch up inputs for
  # efficiency. However, it also means that if we want a prediction
  # for a single instance, we'll need to wrap it in an outer list.
  inputs = {"x": tf.placeholder(tf.float32,shape=[28, 28])}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)
predictions = predict_fn({'x': img})

predict = predictor.from_estimator(predictions, serving_input_receiver_fn = serving_input_rc_fn)
print(predictions)