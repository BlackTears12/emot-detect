import cv2
import numpy as np
from src import data
from src import model


def get_model(dataset_train, use_saved=False):
    rnn_model = model.Model()
    if use_saved:
        rnn_model.load()
    else:
        rnn_model.train(dataset_train)
    return rnn_model


if __name__ == '__main__':
    batch_size = 40
    dataset_train, dataset_test, dataset_val = data.load_crema_dataset(batch_size=batch_size,samples=10)
    rnn_model = get_model(dataset_train, use_saved=False)
    rnn_model.test(dataset_test)