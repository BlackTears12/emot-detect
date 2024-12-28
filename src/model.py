import os.path
import tensorflow as tf
from tensorflow import keras
import data


class Model:
    def __init__(self):
        self.input = keras.Input(shape=(10, data.INPUT_SHAPE[0], data.INPUT_SHAPE[1], data.INPUT_SHAPE[2]))
        self.pretrained = tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=data.INPUT_SHAPE)
        self.pretrained.trainable = False
        self.feature_extractor = tf.keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(filters=16, kernel_size=(4, 4), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(2, 2)),
            keras.layers.Conv2D(filters=64, kernel_size=(6, 6), activation='relu', padding='same'),
            keras.layers.MaxPooling2D(pool_size=(5, 5)),
            keras.layers.Flatten()
        ])
        x = self.pretrained.input
        x = self.feature_extractor(x)
        self.cnn_model = keras.Model(inputs=self.pretrained.inputs, outputs=x)

        self.classifier = tf.keras.Sequential([
            keras.layers.TimeDistributed(self.cnn_model),
            keras.layers.Masking(),
            keras.layers.LSTM(64, activation='relu', return_sequences=False,use_bias=True),
            keras.layers.Dense(128, activation='linear'),
            keras.layers.Dense(len(data.EMOTIONS), activation='softmax'),
        ])

        x = self.classifier(self.input)
        self.model = keras.Model(inputs=self.input, outputs=x)
        self.model.summary()
        self.save_file = "rnn_model.weights.h5"

    def compile(self):
        self.model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, dataset_train, n_epochs=10, batch_size=32):
        self.compile()
        self.model.fit(dataset_train, epochs=n_epochs, batch_size=batch_size)
        self.save()

    def test(self, dataset_test):
        b_data, b_label = dataset_test.__getitem__(0)
        pred = self.model.predict(b_data)
        for i in range(10):
            pr,label = pred[i], b_label[i]
            print(pr)
            print(label)

    def save(self):
        self.model.save_weights(self.save_file)

    def load(self):
        if os.path.isfile(self.save_file):
            self.model.load_weights(self.save_file)
