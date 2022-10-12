from DataUtils2 import DataUtils2
import dill
import scipy.io
a=DataUtils2(r"dataset\reg")
dataset=a.create_data()
# print(dataset)
with open(r'dataset\final\\' + "dataset_try" + '.dill', 'wb') as f:
    dill.dump(dataset, f)
# scipy.io.savemat(r'dataset\final\\' + "dataset_try" + '.mat', mdict = {'perf_data': dataset})
