import cv2
import numpy as np
import data
import model
import argparse

def show_first_batch(dataloader):
    f, l = dataset_train.__getitem__(0)
    for i in range(batch_size):
        print(data.decode_emotion(np.argmax(l[i])))
        for img in f[i]:
            cv2.imshow('image', img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='NN optional arguments')
    parser.add_argument('--load-saved', action='store_true',
        help='Use saved model')
    return parser.parse_args().load_saved


if __name__ == '__main__':    
    batch_size = 32
    dataset_train, dataset_test, dataset_val = data.load_crema_dataset(batch_size=batch_size, samples=12)
    rnn_model = model.Model()
    if parse_args():
        rnn_model.load()
    else:
        rnn_model.train(dataset_train,n_epochs=15,batch_size=batch_size)
    rnn_model.test(dataset_test)
