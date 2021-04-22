import os

import joblib
import muspy
import torch
import random
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class RawDataset(object):
    def __init__(self, path, shuffle=True):
        self.path = path
        self.files = []
        self.train_files = []
        self.dev_files = []
        self.test_files = []
        self.train_data = []
        self.dev_data = []
        self.test_data = []
        self.labels = []
        self.shuffle = shuffle

        self.get_filenames()
        self.read_data()

    def get_filenames(self):
        self.labels = os.listdir(self.path)
        for d in self.labels:
            files = os.listdir(os.path.join(self.path, d))
            train, test = train_test_split(files, test_size=0.2, random_state=17)
            train, dev = train_test_split(train, test_size=0.2, random_state=17)
            self.files.extend([os.path.join(d, i) for i in files])
            self.train_files.extend([os.path.join(d, i) for i in train])
            self.dev_files.extend([os.path.join(d, i) for i in dev])
            self.test_files.extend([os.path.join(d, i) for i in test])

        if self.shuffle:
            random.shuffle(self.train_files)
            random.shuffle(self.dev_files)
            random.shuffle(self.test_files)

    def read_data(self):
        filename_lists = [self.train_files, self.dev_files, self.test_files]
        data_lists = [self.train_data, self.dev_data, self.test_data]

        for files, data in zip(filename_lists, data_lists):
            for midi in files:
                label = os.path.split(midi)[0]
                try:
                    music = muspy.read_midi(os.path.join(self.path, midi))
                    # choose the longest track
                    track_len = [len(i) for i in music.tracks]
                    track = music.tracks[track_len.index(max(track_len))]

                    # Note-based representation:
                    # (time, pitch, duration, velocity) for each note, used as 4 channels in ResNet
                    rep = muspy.Music(resolution=music.resolution, tracks=[track]).to_note_representation()
                    data.append((rep, label))
                except Exception as e:
                    print(f'Failed to read file {midi}!')
                    print(e)

    def save(self, filename='data.joblib'):
        joblib.dump(self, filename)

    @staticmethod
    def load(filename='data.joblib'):
        return joblib.load(filename)


class TorchDataset(Dataset):
    def __init__(self, labels, inputs):
        self.labels = labels
        self.inputs = inputs

    def __getitem__(self, item):
        # NOTICE: Type of input is float!
        return torch.tensor(self.labels[item]), torch.tensor(self.inputs[item]).float()

    def __len__(self):
        return len(self.inputs)


def collate_fn(data):
    labels, inputs = map(list, zip(*data))
    labels = torch.tensor(labels)
    inputs = pad_sequence(inputs, batch_first=True)

    # (Batch, Length, Channels) -> (Batch, Channels, Length)
    inputs = inputs.transpose(1, 2)
    return labels, inputs


class DataModule(LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.bath_size = batch_size
        self.data: RawDataset = None
        self.train_dataset: TorchDataset = None
        self.dev_dataset: TorchDataset = None
        self.test_dataset: TorchDataset = None

    def prepare_data(self):
        if os.path.exists(f'{self.path}.joblib'):
            self.data = RawDataset.load(f'{self.path}.joblib')
            print(f'Loaded data from exsiting file({self.path}.joblib)! If the dataset has been changed, DELETE this file!')
        else:
            self.data = RawDataset(self.path, shuffle=True)
            self.data.save(f'{self.path}.joblib')

    def setup(self, stage=None):
        self.train_dataset, self.dev_dataset, self.test_dataset = (
            TorchDataset(
                # string label -> numeric label
                labels=[self.data.labels.index(i[1]) for i in d],
                # scale per sample
                inputs=[scale(i[0]) for i in d]
                # inputs=[np.delete(i[0], 0, axis=1) for i in d]
            ) for d in [self.data.train_data, self.data.dev_data, self.data.test_data])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.bath_size,
            collate_fn=collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dev_dataset,
            batch_size=self.bath_size,
            collate_fn=collate_fn,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.bath_size,
            collate_fn=collate_fn,
            shuffle=False
        )
