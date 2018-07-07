from tensorflow import keras


def define_model(input_shape, num_classes):
        model = keras.models.Sequential()
        model.add(
        keras.layers.Conv2D(filters=4, kernel_size=(2, 2), strides=1, activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(filters=4, kernel_size=(2, 2), strides=1, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(rate=0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=8, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.5))
        model.add(keras.layers.Dense(units=num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model
