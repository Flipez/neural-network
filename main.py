import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.executing_eagerly()

#tf.keras.backend.experimental.disable_tf_random_generator()
#tf.keras.utils.set_random_seed(1)
#tf.config.experimental.enable_op_determinism()

mnist = tf.keras.datasets.mnist

(training_data, training_labels), (test_data, test_labels) = mnist.load_data()


#training_data, test_data = training_data, test_data
training_data, test_data = training_data / 255, test_data / 255

hidden_layer_accuracy = []

#for hidden_layer_size in range(11, 784):
with tf.device('/cpu:0'):
    model = tf.keras.Sequential([ 
        # flatten will line up all pixes from the input images
        tf.keras.layers.Flatten(input_shape=(28,28)),
        
        # best hidden layer size has been calculated by running
        # every hidden layer size between 11 and 784 for 5 epochs
        tf.keras.layers.Dense(756, activation=tf.nn.relu),
        
        # requested size of the output layer
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adamax(),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    print("Fit")
    model.fit(training_data, training_labels, epochs=10, verbose=1)
    
    print("Eval")
    model.evaluate(test_data, test_labels, verbose=1)
    
    predictions = model.predict(test_data, verbose=1)
#        print("{}: {}".format(hidden_layer_size, history.history['accuracy'][0]))

#    hidden_layer_accuracy.append("{}: {}".format(hidden_layer_size, history.history['accuracy'][0]))

#print(hidden_layer_accuracy)

true_predictions = []
false_predictions = []

for idx, prediction in enumerate(predictions):
    if np.argmax(prediction) == test_labels[idx]:
        true_predictions.append(idx)
    else:
        false_predictions.append(idx)

print("Made {} true predictions, for example index {}".format(len(true_predictions), true_predictions[0]))
print("Made {} false predictions, for example index {}".format(len(false_predictions), false_predictions[0]))


plt.title(
    "Real Value: {}\nPrediction Value: {}".format(
        test_labels[true_predictions[0]],
        np.argmax(predictions[true_predictions[0]])
    )
)
plt.imshow(test_data[true_predictions[0]], cmap='Greys')
plt.show()

plt.title(
    "Real Value: {}\nPrediction Value: {}".format(
        test_labels[false_predictions[0]],
        np.argmax(predictions[false_predictions[0]])
    )
)
plt.imshow(test_data[false_predictions[0]], cmap='Greys')
plt.show()