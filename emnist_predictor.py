from tensorflow.contrib import predictor
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
import sys

#prepare image for predictor
def prep_img(filepath):
    #load image
    img = imageio.imread(filepath)
    #convert to numpy array
    img = np.asarray(img,dtype=np.float32)
    #comes in as (28,28,3)
    #assume image is rgb, get rid of 3rd dimension
    #will be shaped: (28,28)
    img = img[:,:,0]
    #normalize from 0-255 to 0-1
    img /= 255
    #wrap our image in a list cause everything in tf assumes a batch
    #should now be (1,28,28)
    img = np.expand_dims(img,0)
    return img

#image must be shaped (-1, 28, 28)
def get_predictions(image, modeldir):
    predict_fn = predictor.from_saved_model(modeldir, signature_def_key='predict')
    predictions = predict_fn({"x": image})
    return predictions

img = prep_img('Images/four.png')
pred = get_predictions(img,'./tmp/saved_model/1523659560')
print(pred["class_ids"])
