import gzip
import numpy as np
import tensorflow as tf
from scipy import io as spio
import matplotlib.pyplot as plt

emnist = spio.loadmat('datasets/emnist/emnist-byclass')

#must be 3d array (ie. batch of images)
def flipandrot(array):
  index = 0
  for inner_array in array:
    array[index] = np.fliplr(inner_array)
    array[index] = np.rot90(inner_array)
    index += 1
  return array

# load training data
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)
x_train = x_train.reshape(x_train.shape[0], 28, 28)


# load training labels
# need not one-hot encoding for DNNClassifier, instead int32
y_train = emnist["dataset"][0][0][0][0][0][1]
y_train = y_train.astype(np.int32)

# load test data
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], 28, 28)


# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]
y_test = y_test.astype(np.int32)

#x_norm  = (x_train - (255 / 2.0)) / 255
#normalize
x_train /= 255
x_test /= 255

#flip and rotate 90 degrees
x_train = flipandrot(x_train)
x_test = flipandrot(x_test)

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28,28])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 32],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=62,
    dropout=0.1,
    model_dir="./tmp/mnist_model"
)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_train},
    y=y_train,
    num_epochs=None,
    batch_size=128,
    shuffle=True
)

print('Training...')
classifier.train(input_fn=train_input_fn, steps=100000)
print('Done')

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_test},
    y=y_test,
    num_epochs=1,
    shuffle=False
)

# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

ft = {"x": tf.placeholder(tf.float32, shape=[None,28,28])}
rawfn = tf.estimator.export.build_raw_serving_input_receiver_fn(ft)

export_dir_base = './tmp/saved_model'
modeldir = classifier.export_savedmodel(export_dir_base, rawfn)


print(modeldir)



