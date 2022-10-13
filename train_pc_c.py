from __future__ import print_function
import gc
import os
import sys
import math
import time
import scipy.stats as ss
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.PCConvNet import PCConvNet, PCConvNetCls, PCConvNetContrastive, PCConvNetContrastive2
from models.PCConvLstmNet import PCConvLstmNet, PCConvLstmNetCls
from dataLoaders.audiodataset import audioDataset
from dataLoaders.audioDataloader import AudioDataLoader
from tensorboard_logger import configure, log_value
from sklearn import metrics
import eval_utils
import train_utils
import dill
from contrastive_utils import ContrastiveLoss
from tqdm import tqdm
import matplotlib.pyplot as plt



# set manual random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)
# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)
if CUDA_AVAILABLE != True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# initialize training parameters
RUN = 110
NUM_EPOCHS = 300  # 2000
NUM_CLASSIFIER_EPOCH = 500
NUM_BATCHES = 30
SEGMENT = '2'
MTYPE = 'conv'
CTYPE = 0
# initialize dataset, dataloader and created batched data
NAME = "piano"
# SET CONSTANTS
instrument = 'ALL'
cross_instrument = 'ALL'
experiment = 'paper-results'
METRIC = 2  # 0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality

ADD_NOISE_TEST = False
ADD_NOISE_VALID = False

NOISE_SHAPE = 'triangular'  # triangular, normal, or uniform
INPUT_REP = 'Strumpet'

# SET TRAINING CONSTANTS
split_train = False
earlystop = True
contrastive = True
Skip_encoder = False
MSE_LOSS_STR = 1
CONTR_LOSS_STR = 0
num_labels = 4
margin = 0.5
classification = False
model_type = 'reg' if not classification else str(num_labels) + 'class'
# HYPER-PARAMETERS
LR_RATE = 0.005  # 0.01
LR_RATE_enc = 0.005
W_DECAY = 1e-5  # 5e-4 #1e-5
MOMENTUM = 0.9

# with open(r"dataset/final/dataset_try.dill")as f:
# NUM_DATA_POINTS = len(dill.load(f))
NUM_DATA_POINTS = 800
dataset = audioDataset(r"dataset/final/dataset_try.dill")
dataloader = AudioDataLoader(dataset, NUM_DATA_POINTS, NUM_BATCHES)

tr1, v1, vef, te1, tef = dataloader.create_split_data(300, 250)  # 1000, 500 | 1500, 500 | 2000, 1000
tr2, v2, _, te2, _ = dataloader.create_split_data(500, 250)
tr3, v3, _, te3, _ = dataloader.create_split_data(1000, 500)
# tr4, v4, _, te4, _ = dataloader.create_split_data(2500, 1000)
# tr5, v5, _, te5, _ = dataloader.create_split_data(3000, 1500)
# tr6, v6, vef, te6, tef = dataloader.create_split_data(4000, 2000)
training_data = tr1 + tr2 + tr3  # + tr2 + tr3 #+ tr4 + tr5 + tr6     # this is the proper training data split
validation_data = vef  # + v2 + v3 + v4 + v5 + v6
testing_data = te1 + te2 + te3  # + te4 + te5 + te6

## augment data
# aug_training_data = train_utils.augment_data(training_data)
# aug_training_data = train_utils.augment_data(aug_training_data)
# aug_validation_data = validation_data  #train_utils.augment_data(validation_data)

USAGE = "cla-"
## initialize model
if MTYPE == 'conv':
    if USAGE == 'cla':
        perf_model = PCConvNetCls(1)
    else:
        perf_model = PCConvNetContrastive2(0, num_classes=num_labels, regression=not classification)
elif MTYPE == 'lstm':
    if USAGE == 'cla':
        perf_model = PCConvLstmNetCls()
    else:
        perf_model = PCConvLstmNet()
if torch.cuda.is_available():
    perf_model.cuda()
if USAGE == 'cla':
    criterion = nn.CrossEntropyLoss()
else:
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
if contrastive:
    criterion_contrastive = ContrastiveLoss(margin=margin, num_labels=num_labels)
    if classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MSELoss()
else:
    criterion_contrastive = None
perf_optimizer = optim.SGD(perf_model.parameters(), lr=LR_RATE, momentum=MOMENTUM, weight_decay=W_DECAY)
# perf_optimizer = optim.Adam(perf_model.parameters(), lr = LR_RATE, weight_decay = W_DECAY)
print(perf_model)

# declare save file name
file_info = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + str(METRIC) + '_' + str(
    RUN) + '_' + MTYPE + '_onlyATest'

# configure tensor-board logger
configure('pc_contrastive_runs/' + NAME + '_Reg', flush_secs=2)

## define training parameters
PRINT_EVERY = 1
ADJUST_EVERY = 1000
START = time.time()
# best_val_loss = 1.0
best_loss_contrastive_val = float('inf')
best_valrsq = .20
best_epoch = 0
best_ce_loss_val = float('inf')
# train and validate
losses1=[]
losses2=[]
try:
    print("Training Encoder for %d epochs..." % NUM_EPOCHS)
    if not Skip_encoder:
        for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
            # perform training and validation
            train_loss, train_r_sq, train_accu, train_accu2, val_loss, val_r_sq, val_accu, val_accu2 = train_utils.train_and_validate(
                perf_model, criterion, perf_optimizer, training_data, validation_data, METRIC, MTYPE, CTYPE,
                contrastive=criterion_contrastive)
            if contrastive:
                print('Evaluating')
                # print(eval_utils.eval_acc_contrastive(perf_model, criterion_contrastive, vef, METRIC, MTYPE, CTYPE, criterion_CE=criterion))
                loss_contrastive_val, acc_contrastive_val, ce_loss_val = eval_utils.eval_acc_contrastive(perf_model,
                                                                                                         criterion_contrastive,
                                                                                                         vef, METRIC,
                                                                                                         MTYPE, CTYPE)
                loss_contrastive_train, acc_contrastive_train, ce_loss_train = eval_utils.eval_acc_contrastive(
                    perf_model, criterion_contrastive, training_data, METRIC, MTYPE, CTYPE)
            # adjut learning rate
            # train_utils.adjust_learning_rate(perf_optimizer, epoch, ADJUST_EVERY)
            # log data for visualization later0
            '''
            else:
            ####
                log_value('train_loss', train_loss, epoch)
                log_value('val_loss', val_loss, epoch)
                log_value('train_r_sq', train_r_sq, epoch)
                log_value('val_r_sq', val_r_sq, epoch)
                log_value('train_accu', train_accu, epoch)
                log_value('val_accu', val_accu, epoch)
                log_value('train_accu2', train_accu2, epoch)
                log_value('val_accu2', val_accu2, epoch)
                '''
            if contrastive:
                log_value('Validation contrastive loss', loss_contrastive_val, epoch)
                # log_value('Validation CrossEntropy Loss', ce_loss_val, epoch)
                # log_value('Validation acc', acc_contrastive_val, epoch)
                log_value('Training contrastive loss', loss_contrastive_train, epoch)
                # log_value('Training CrossEntropy Loss', ce_loss_train, epoch)
                # log_value('Training acc', acc_contrastive_train, epoch)
            #####

            # print loss
            if epoch % PRINT_EVERY == 0:
                print('[%s (%d %.1f%%)]' % (train_utils.time_since(START), epoch, float(epoch) / NUM_EPOCHS * 100))
                # print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Train Loss: ', train_loss, ' R-sq: ', train_r_sq, ' Accu:', train_accu, train_accu2))
                # print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))

                if contrastive:
                    # print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Train Contrastive Loss: ', loss_contrastive_train,'Train CE Loss:', ce_loss_train, 'Train Accuracy: ', acc_contrastive_train))
                    # print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Validation Contrastive Loss: ', loss_contrastive_val,'Validation CE Loss:', ce_loss_val, 'Validation Accuracy: ', acc_contrastive_val))
                    print('[%s %0.5f]' % ('Train Contrastive Loss: ', loss_contrastive_train))
                    print('[%s %0.5f]' % ('Validation Contrastive Loss: ', loss_contrastive_val))
            # save model if best validation accuracy
            if loss_contrastive_val.item() < best_loss_contrastive_val:  # acc_contrastive_val > best_acc_contrastive_val: #
                n = 'pc_contrastive_runs/' + NAME + '_best'
                train_utils.save(n, perf_model)
                # best_acc_contrastive_val = acc_contrastive_val
                best_loss_contrastive_val = loss_contrastive_val.item()
                best_epoch = epoch
            # store the best r-squared value from training
            if val_r_sq > best_valrsq:
                best_valrsq = val_r_sq
            if best_epoch < epoch - 250 and earlystop:
                break

        train_utils.save('pc_contrastive_runs/' + NAME, perf_model)
    else:
        try:
            perf_model.load_state_dict(torch.load('pc_contrastive_runs/' + NAME))
            print("load success")
        except:
            pass
    print("Training Classifier for %d epochs..." % NUM_CLASSIFIER_EPOCH)
    if split_train:
        for param in perf_model.conv.parameters():
            param.requires_grad = False

    perf_optimizer = optim.SGD(perf_model.parameters(), lr=LR_RATE_enc, momentum=MOMENTUM, weight_decay=W_DECAY)
    for epoch in tqdm(range(1, NUM_CLASSIFIER_EPOCH + 1)):
        # perform training and validation
        train_loss, train_r_sq, train_accu, train_accu2, val_loss, val_r_sq, val_accu, val_accu2 = train_utils.train_and_validate(
            perf_model, criterion, perf_optimizer, training_data, validation_data, METRIC, MTYPE, CTYPE,
            contrastive=criterion_contrastive, encoder=False, classification= not classification)
        if not classification:
            loss_contrastive_val, acc_contrastive_val, ce_loss_val = eval_utils.eval_acc_contrastive(perf_model,
                                                                                                     criterion_contrastive,
                                                                                                     vef, METRIC, MTYPE,
                                                                                                     CTYPE,
                                                                                                     criterion_CE=criterion)
            loss_contrastive_train, acc_contrastive_train, ce_loss_train = eval_utils.eval_acc_contrastive(perf_model,
                                                                                                           criterion_contrastive,
                                                                                                           training_data,
                                                                                                        METRIC,
                                                                                                        MTYPE, CTYPE,
                                                                                                     criterion_CE=criterion)
            losses1.append(ce_loss_val.cpu())
            losses2.append(ce_loss_train.cpu())
        # adjut learning rate
        # train_utils.adjust_learning_rate(perf_optimizer, epoch, ADJUST_EVERY)
        # log data for visualization later
        if classification:
            ####
            log_value('train_loss', train_loss, epoch)
            log_value('val_loss', val_loss, epoch)
            log_value('train_r_sq', train_r_sq, epoch)
            log_value('val_r_sq', val_r_sq, epoch)
            log_value('train_accu', train_accu, epoch)
            log_value('val_accu', val_accu, epoch)
            log_value('train_accu2', train_accu2, epoch)
            log_value('val_accu2', val_accu2, epoch)
        else:
            # log_value('Validation contrastive loss', loss_contrastive_val, epoch)
            log_value('Validation CrossEntropy Loss', ce_loss_val, epoch)
            log_value('Validation acc', acc_contrastive_val, epoch)
            # log_value('Training contrastive loss', loss_contrastive_train, epoch)
            log_value('Training CrossEntropy Loss', ce_loss_train, epoch)
            log_value('Training acc', acc_contrastive_train, epoch)
        #####

        # print loss
        if epoch % PRINT_EVERY == 0:
            print('[%s (%d %.1f%%)]' % (train_utils.time_since(START), epoch, float(epoch) / NUM_EPOCHS * 100))
            # print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Train Loss: ', train_loss, ' R-sq: ', train_r_sq, ' Accu:', train_accu, train_accu2))
            # print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))

            if not classification:
                # print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Train Contrastive Loss: ', loss_contrastive_train,'Train CE Loss:', ce_loss_train, 'Train Accuracy: ', acc_contrastive_train))
                # print('[%s %0.5f, %s %0.5f, %s %0.5f]'%('Validation Contrastive Loss: ', loss_contrastive_val,'Validation CE Loss:', ce_loss_val, 'Validation Accuracy: ', acc_contrastive_val))
                print('[%s %0.5f, %s %0.5f]' % (
                    'Train CE Loss:', ce_loss_train, 'Train Accuracy: ', acc_contrastive_train))
                print('[%s %0.5f, %s %0.5f]' % (
                    'Validation CE Loss:', ce_loss_val, 'Validation Accuracy: ', acc_contrastive_val))
            else:
                print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]' % (
                    'Train Loss: ', train_loss, ' R-sq: ', train_r_sq, ' Accu:', train_accu, train_accu2))
                print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]' % (
                    'Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))
        # save model if best validation accuracy
        loss_epoch = ce_loss_val if not classification else val_loss
        if loss_epoch < best_ce_loss_val:  # loss_contrastive_val.item() < best_loss_contrastive_val:
            n = 'pc_contrastive_runs/' + NAME + '_best'
            train_utils.save(n, perf_model)
            best_ce_loss_val = loss_epoch
            # best_loss_contrastive_val = loss_contrastive_val.item()
            best_epoch = epoch
        # store the best r-squared value from training
        # if val_r_sq > best_valrsq:
        #     best_valrsq = val_r_sq
        # if best_epoch < epoch - 100 and earlystop:
        #     break
    print("Saving...")
    train_utils.save('pc_contrastive_runs/' + NAME, perf_model)
except KeyboardInterrupt:
    print("Saving before quit...")
    train_utils.save('pc_contrastive_runs/' + NAME, perf_model)
if classification:
    print('BEST Accuracy: ' + str(acc_contrastive_val.data))


else:
    print('BEST R^2 VALUE: ' + str(best_valrsq))

# test
# test of full length data
if contrastive:
    loss_contrastive_test, acc_contrastive_test, ce_loss_test = eval_utils.eval_acc_contrastive(perf_model,
                                                                                                criterion_contrastive,
                                                                                                vef, METRIC, MTYPE,
                                                                                                CTYPE,
                                                                                                criterion_CE=criterion)
    print('[%s %0.5f, %s %0.5f, %s %0.5f]' % (
        'Testing Contrastive Loss: ', loss_contrastive_test, 'Testing CE Loss:', ce_loss_test, 'Testing Accuracy: ',
        acc_contrastive_test))
else:
    test_loss, test_r_sq, test_accu, test_accu2 = eval_utils.eval_model(perf_model, criterion, testing_data, METRIC,
                                                                        MTYPE, CTYPE)
    print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]' % (
        'Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))

# validate and test on best validation model
# read the model
# filename = file_info + '_Reg'
# filename = NAME + '_best'
filename = NAME
if torch.cuda.is_available():
    perf_model.cuda()
    # perf_model.load_state_dict(torch.load('/Users/michaelfarren/Desktop/MusicPerfAssessment-claer/src/runs/' + filename + '.pt'))
    perf_model.load_state_dict(torch.load('pc_contrastive_runs/' + filename))
else:
    perf_model.load_state_dict(
        torch.load('pc_contrastive_runs/' + filename + '.pt', map_location=lambda storage, loc: storage))

if contrastive and not classification:
    loss_contrastive_val, acc_contrastive_val, ce_loss_val = eval_utils.eval_acc_contrastive(perf_model,
                                                                                             criterion_contrastive, vef,
                                                                                             METRIC, MTYPE, CTYPE,
                                                                                             criterion_CE=criterion)
    print('[%s %0.5f, %s %0.5f, %s %0.5f]' % (
        'Validation Contrastive Loss: ', loss_contrastive_val, 'Validation CE Loss:', ce_loss_val,
        'Validation Accuracy: ',
        acc_contrastive_val))
else:
    val_loss, val_r_sq, val_accu, val_accu2 = eval_utils.eval_model(perf_model, criterion, vef, METRIC, MTYPE, CTYPE)
    print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]' % (
        'Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))

plt.plot(range(len(losses1)),losses1,label="val_loss")
plt.plot(range(len(losses2)),losses2,label="train_loss")
plt.legend()
plt.show()


if contrastive and not classification:
    loss_contrastive_test, acc_contrastive_test, ce_loss_test = eval_utils.eval_acc_contrastive(perf_model,
                                                                                                criterion_contrastive,
                                                                                                testing_data, METRIC,
                                                                                                MTYPE, CTYPE,
                                                                                                criterion_CE=criterion)
    print('Finally:[%s %0.5f, %s %0.5f, %s %0.5f]' % (
        'Testing Contrastive Loss: ', loss_contrastive_test, 'Testing CE Loss:', ce_loss_test, 'Testing Accuracy: ',
        acc_contrastive_test))
else:
    test_loss, test_r_sq, test_accu, test_accu2 = eval_utils.eval_model(perf_model, criterion, testing_data, METRIC,
                                                                        MTYPE, CTYPE)
    print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]' % (
        'Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))
