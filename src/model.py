import os.path
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import numpy as np
import data


class Model:
    def __init__(self):
        self.cnn_input = keras.layers.Input(shape=data.INPUT_SHAPE)
        self.pretrained = tf.keras.applications.InceptionResNetV2(include_top=False)
        self.pretrained.trainable = False
        self.feature_extractor = tf.keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),            
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),            
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same'),
            keras.layers.Flatten()
        ])
        x = self.cnn_input
        x = self.pretrained(x)
        x = self.feature_extractor(x)
        self.cnn_model = keras.Model(inputs=self.cnn_input, outputs=x)

        self.classifier = tf.keras.Sequential([
            keras.layers.Input(shape=(None,data.INPUT_SHAPE[0],data.INPUT_SHAPE[1],data.INPUT_SHAPE[2])),
            keras.layers.TimeDistributed(self.cnn_model),
            keras.layers.LSTM(10, activation='relu', return_sequences=False,use_bias=True),
            keras.layers.Dense(16, activation='linear'),
            keras.layers.Dense(len(data.EMOTIONS), activation='softmax'),
        ])

        self.model = self.classifier
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
        test_acc = []
        for i in range(len(pred)):
            pr,label = pred[i], b_label[i]
            test_acc.append(np.argmax(pr) == np.argmax(label))
        good_preds = np.count_nonzero(np.array(test_acc))
        print("Test prediction accuracy",good_preds/len(test_acc))
            

    def save(self):
        self.model.save_weights(self.save_file)
        self.model.save_weights("weights/"+datetime.now().strftime("%d-%m-%Y-%H:%M")+self.save_file)

    def load(self):
        if os.path.isfile(self.save_file):
            self.model.load_weights(self.save_file)
