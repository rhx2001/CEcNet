import sys
import os
import json
import dill
from collections import defaultdict
import numpy as np
import pandas as pd
from pandas import ExcelFile
import librosa


class DataUtils2(object):

    def __init__(self, path):
        self.path = path
        self.id = []
        self.emo = []
        self.name_set = os.listdir(self.path)

    def scan_file_id(self):
        id_set = []
        for i in self.name_set:
            i = i[3:-4]
            id_set.append(i)
        return id_set

    def scan_emo_id(self):
        emo_set = []
        for i in self.name_set:
            i = i[:2]
            emo_set.append(i)
        return emo_set

    def create_label(self):
        label_set = []
        for j, i in enumerate(self.name_set):
            label_set.append(j)
        return label_set

    def create_data(self):
        # self=DataUtils(r"dataset\reg")
        id = self.scan_file_id()
        emo = self.scan_emo_id()
        label = self.create_label()
        pref_data = []
        for idx in range(len(id)):
            as_data = {}
            as_data["id"] = id[idx]
            as_data["emo"] = emo[idx]
            as_data["label"] = label[idx]
            print(".\\" + self.path + "\\" + self.name_set[idx])
            y, sr = librosa.load(".\\" + self.path + "\\" + self.name_set[idx])
            as_data["audio"] = librosa.feature.mfcc(y, sr=44100)
            pref_data.append(as_data)
        return pref_data


if __name__ == "__main__":
    a = DataUtils2(r"dataset\reg2")
    t = a.create_data()
    print(t)
