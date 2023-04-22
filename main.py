import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.executing_eagerly()

mnist = tf.keras.datasets.mnist

(training_data, training_labels), (test_data, test_labels) = mnist.load_data()


#training_data, test_data = training_data, test_data
training_data, test_data = training_data / 255, test_data / 255

hidden_layer_accuracy = []

for hidden_layer_size in range(11, 784):
    with tf.device('/cpu:0'):
        model = tf.keras.Sequential([ 
            tf.keras.layers.Flatten(input_shape=(28,28)),        # flatten = bildpixel alle aneinander reihen
            tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu),    # erste hidden layer
            tf.keras.layers.Dense(10, activation=tf.nn.softmax), # Ausgabe Layer
        ])
    
        model.compile(
            optimizer = tf.keras.optimizers.Adamax(),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
        )
        
        history = model.fit(training_data, training_labels, epochs=5, batch_size=80, verbose=0)
        
        model.evaluate(test_data, test_labels, verbose=0)
        
        predictions = model.predict(test_data, verbose=0)
        print("{}: {}".format(hidden_layer_size, history.history['accuracy'][0]))

    hidden_layer_accuracy.append("{}: {}".format(hidden_layer_size, history.history['accuracy'][0]))

print(hidden_layer_accuracy)
image_index = 700


plt.title(
    "Real Value: {}\nPrediction Value: {}".format(
        test_labels[image_index],
        np.argmax(predictions[image_index])
    )
    #f'Echter Wert: {test_labels[image_index]} \nVorhersage-Wert: (0-9): {np.argmax(predictions[image_index])}'
)
plt.imshow(test_data[image_index], cmap='Greys')
plt.show()