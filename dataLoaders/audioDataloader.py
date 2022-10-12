import os
import sys
import collections
from typing import List

import numpy as np
import random
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as multiprocessing
from dataLoaders.audiodataset import audioDataset


class AudioDataLoader(DataLoader):

    def __init__(self, dataset, num_data_pts, num_batches=1):
        # super().__init__(dataset)
        legnth = len(dataset)
        self.dataset = dataset
        self.num_data_pts = num_data_pts
        self.num_batches = num_batches
        self.indices = np.arange(self.num_data_pts)
        np.random.shuffle(self.indices)
        self.mini_batch_size = int(np.floor(self.num_data_pts / self.num_batches))

    def create_split_data(self, chunk_len, hop):
        """

        Args:
            chunk_len: The size of one segment
            hop: the gap of pairs of segment

        Returns:batched_data, val_batch, val_batch_full, test_batch, test_batch_full

        """

        full_datas = []
        random.seed(0)
        indices = self.indices
        num_training_songs = int(0.8 * self.num_data_pts)
        num_validation_songs = int(0.1 * self.num_data_pts)
        num_testing_songs = num_validation_songs
        num_songs = [num_training_songs, num_validation_songs, num_testing_songs]
        sums: list[int] = [0] * 3
        sums[0] = num_training_songs
        for i in range(1, 3):
            sums[i] = num_songs[i] + sums[i - 1]

        sums = [0] + sums
        for j in range(len(sums) - 1):
            print(j)
            train_split = []
            full_batch = []
            for i in range(sums[j], sums[j + 1]):
                data = self.dataset[indices[i]]
                pc = data['audio']
                gt = data['emo']
                count = 0
                if len(pc[0]) < chunk_len:
                    zeropad_pc = np.zeros((pc.shape[0], chunk_len))
                    for k in range(len(pc)):
                        zeropad_pc[k][:pc.shape[1], ] = pc[k]
                    pc = zeropad_pc
                while count + chunk_len <= len(pc[0]):
                    d = {}
                    d['audio'] = pc[:, count: count + chunk_len]
                    d['emo'] = gt
                    d["label"] = np.array([data["label"]])
                    train_split.append(d)
                    full_batch.append(d)
                    count += hop
            shuffle(train_split)

            num_data_pts = len(train_split)
            batched_data = [None] * self.num_batches
            mini_batch_size = int(np.floor(num_data_pts / self.num_batches))
            count = 0
            for batch_num in range(self.num_batches):
                batched_data[batch_num] = list()
                audio_tensor = torch.zeros(mini_batch_size, 20, chunk_len)
                emo_tensor = torch.zeros(mini_batch_size, len(train_split[count]['emo']))
                label_tensor = torch.zeros(mini_batch_size, len(train_split[count]['label']))
                for seq_num in range(mini_batch_size):
                    # convert pitch contour to torch tensor
                    au_tensor = torch.from_numpy(train_split[count]['audio'])
                    audio_tensor[seq_num, :, :] = au_tensor.float()
                    # convert score tuple to torch tensor
                    s_tensor = torch.from_numpy(np.asarray(train_split[count]['emo']))
                    emo_tensor[seq_num, :] = s_tensor
                    label_tensor[seq_num, :] = torch.from_numpy(train_split[count]["label"])
                    count += 1
                label_tensor += 1
                dummy = {}
                dummy['audio'] = audio_tensor
                dummy['emo'] = emo_tensor
                dummy['label'] = label_tensor
                batched_data[batch_num] = dummy
            if j != 0:
                full_datas.append(batched_data)
                full_datas.append(full_batch)
            else:
                full_datas.append(batched_data)
        return full_datas


if __name__ == "__main__":
    t = audioDataset(r"dataset/final/dataset_try.dill")
    a = AudioDataLoader(t, len(t), 10)
    b, c, d, e, f = a.create_split_data(1000, 500)
    print(b[0]["audio"].shape)
    print(b[0]["emo"].shape)
    print(b[0]["label"].shape)
    print(b[0])
