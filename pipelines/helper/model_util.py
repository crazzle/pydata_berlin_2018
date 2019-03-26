import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf

def define_model(input_shape, num_classes):
        model = Sequential()
        model.add(keras.layers.Conv2D(filters=4, kernel_size=(2, 2),
                                strides=1, activation='relu',
                                input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Conv2D(filters=4, kernel_size=(2, 2),
                                strides=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=num_classes, activation= (tf.nn.softmax)))
        model.compile(loss = 'categorical_crossentropy',
              optimizer= keras.optimizers.RMSprop(),
              metrics= ['accuracy'])
        return model