import os
import dill
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class audioDataset(Dataset):

    def __init__(self,data_path):

        '''
        data{
            id:name of piceof data
            emo:type of emotion
            label:used to identify positive or negative
            audio:mfcc/cqt,,,,
        }
        '''

        self.perf_data = dill.load(open(data_path, 'rb'))
        #print(self.perf_data[0])
        # print(len(self.perf_dat))
        self.length = len(self.perf_data)
        for j,i in enumerate(self.perf_data):
            self.perf_data[j]["emo"]=self.one_hot(i["emo"])

    def __getitem__(self,i):
        return self.perf_data[i]

    def __len__(self):
        return self.length

    @staticmethod
    def one_hot(y):
        emo_dict={
            "Q1": 0,
            "Q2": 1,
            "Q3": 2,
            "Q4": 3,
        }
        tmp=np.array([0]*4).T
        tmp[emo_dict[y]]=1
        # print(tmp.shape)
        return tmp

if __name__ == "__main__":
    a=audioDataset(r"dataset/final/dataset_try.dill")
    print(a.__len__())
    print(a)
    # a[0]["audio"].shape
