import asyncio
import csv
import math
import numpy as np
import cv2 as cv
import os

from tensorflow.keras.utils import Sequence
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

FRAME_DIMS = (224, 224)
INPUT_SHAPE = (FRAME_DIMS[0], FRAME_DIMS[1], 3)

EMOTIONS = ["SAD", "HAP", "FEA", "ANG", "DIS", "NEU"]


def encode_emotion(emotion: str):
    return EMOTIONS.index(emotion)


def decode_emotion(code: int):
    return EMOTIONS[code]


def __get_label(filename):
    emotion = filename.split('_')[2]
    return encode_emotion(emotion)


def __parse_summary(filenames_file):
    with open(filenames_file) as f:
        next(f)
        reader = csv.reader(f)
        filenames = [s[1] for s in reader if "SAD" in s[1] or "ANG" in s[1]]
        return (np.array([f + ".mp4" for f in filenames]),
                np.array([__get_label(f) for f in filenames]))


def load_crema_dataset(summary_file="dataset/CREMA-D/SentenceFilenames.csv",
                       video_dir="dataset/CREMA-D/Extracted",
                       samples=10,
                       split_test=0.2, split_val=0.1,
                       batch_size=32) -> (Dataset, Dataset, Dataset):
    files, labels = __parse_summary(summary_file)
    shuffler = np.random.permutation(len(files))
    files = files[shuffler]
    labels = labels[shuffler]
    test_beg = math.floor((1 - split_test) * len(files)) + 1
    val_beg = test_beg + math.floor(split_val * len(files)) + 1
    train = files[:test_beg], labels[:test_beg]
    test = files[test_beg:val_beg], labels[test_beg:val_beg]
    val = files[val_beg:], labels[val_beg:]
    return (CremaDataloader(train[0], train[1], video_dir, samples,batch_size),
            CremaDataloader(test[0], test[1], video_dir, samples,batch_size),
            CremaDataloader(val[0], val[1], video_dir, samples,batch_size))


class CremaDataloader(Sequence):
    def __init__(self, videos, labels, video_dir, samples, batch_size):
        super().__init__()
        self.videos = videos
        self.labels = labels
        self.video_dir = video_dir
        self.samples = samples
        self.batch_size = batch_size
        self.cache = None
        self.cache_ind = -1

    def __len__(self):
        return math.floor(len(self.videos) / self.batch_size)

    def __getitem__(self, idx):
        frames = self.__load_datafiles(self.videos[idx * self.batch_size:(idx + 1) * self.batch_size])
        labels = np.zeros((self.batch_size,len(EMOTIONS)))
        ind = 0
        for label in self.labels[idx*self.batch_size:(idx+1)*self.batch_size]:
            labels[ind][label] = 1
            ind += 1
        return frames, labels

    def __extract_frames_from_video(self,video_path) -> np.ndarray:
        output = []
        cap = cv.VideoCapture(video_path)
        success, frame = cap.read()
        while success:
            output.append(frame)
            success, frame = cap.read()
        output = np.array(output)
        indices = np.linspace(0, len(output) - 1, self.samples, dtype=int)
        return np.divide(output[indices],255)

    def __load_datafiles(self, filenames):
        return np.array([
            self.__extract_frames_from_video(os.path.join(self.video_dir,file)) for file in filenames])
