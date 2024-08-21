from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
import pickle
from collections import Counter

seed = 10
data_x = np.load(r'E:\competition\睡眠呼吸暂停和低通气事件检测\data\训练集/train_x.npy')
data_y_array = np.load(r'E:\competition\睡眠呼吸暂停和低通气事件检测\data\训练集/train_y.npy')

group_list = []

for i in range(len(data_x)):
    if i == 0:
        group_one = [i]
        continue
    if i == (len(data_x) - 1):
        group_one.append(i)
        group_list.append(group_one)
        continue
    signal1_last = data_x[i, 0, -1]
    signal2_last = data_x[i, 1, -1]
    signal1_next = data_x[i+1, 0, 0]
    signal2_next = data_x[i+1, 1, 0]
    if abs(signal1_last - signal1_next) > 2 or abs(signal2_last - signal2_next) > 10:
        group_list.append(group_one)
        group_one = [i]
    else:
        group_one.append(i)

x_index = [_ for _ in range(len(group_list))]
data_y = [0 for _ in range(len(group_list))]

trainval_test_sfolder = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=seed)
train_val_sfolder = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)

split_data = []

for trainval_index, test_index in trainval_test_sfolder.split(x_index, data_y):
    train_ID = []
    val_ID = []
    test_ID = []

    trainval_ID = []
    trainval_label = []
    for id in trainval_index:
        trainval_ID.append(x_index[id])
        trainval_label.append(data_y[id])
    for id in test_index:
        test_ID.append(x_index[id])

    for train_index, val_index in train_val_sfolder.split(trainval_ID, trainval_label):
        for id in train_index:
            train_ID.append(trainval_ID[id])
        for id in val_index:
            val_ID.append(trainval_ID[id])

    train_ID_real = []
    val_ID_real = []
    test_ID_real = []
    for id in train_ID:
        train_ID_real.extend(group_list[id])
    for id in val_ID:
        val_ID_real.extend(group_list[id])
    for id in test_ID:
        test_ID_real.extend(group_list[id])
    print(train_ID_real, val_ID_real, test_ID_real)
    split_data.append({'train': train_ID_real, 'val': val_ID_real, 'test': test_ID_real})

split_data_new = []
for i, split_data in enumerate(split_data):
    train_ID = split_data['train']
    val_ID = split_data['val']
    test_ID = split_data['test']
    val_label = data_y_array[split_data['val']]
    test_label = data_y_array[split_data['test']]
    val_num = min(np.sum(val_label == 1), np.sum(val_label == 2))
    test_num = min(np.sum(test_label == 1), np.sum(test_label == 2))

    val_num_0 = val_num
    val_num_1 = val_num
    val_num_2 = val_num
    val_ID_new = []
    for id in val_ID:
        if data_y_array[id] == 0:
            if val_num_0 > 0:
                val_ID_new.append(id)
            else:
                train_ID.append(id)
            val_num_0 -= 1
        if data_y_array[id] == 1:
            if val_num_1 > 0:
                val_ID_new.append(id)
            else:
                train_ID.append(id)
            val_num_1 -= 1
        if data_y_array[id] == 2:
            if val_num_2 > 0:
                val_ID_new.append(id)
            else:
                train_ID.append(id)
            val_num_2 -= 1

    test_num_0 = test_num
    test_num_1 = test_num
    test_num_2 = test_num
    test_ID_new = []
    for id in test_ID:
        if data_y_array[id] == 0:
            if test_num_0 > 0:
                test_ID_new.append(id)
            else:
                train_ID.append(id)
            test_num_0 -= 1
        if data_y_array[id] == 1:
            if test_num_1 > 0:
                test_ID_new.append(id)
            else:
                train_ID.append(id)
            test_num_1 -= 1
        if data_y_array[id] == 2:
            if test_num_2 > 0:
                test_ID_new.append(id)
            else:
                train_ID.append(id)
            test_num_2 -= 1

    split_data_new.append({'train': train_ID, 'val': val_ID_new, 'test': test_ID_new})

with open('train_val_test_5folds_seed{}.pkl'.format(seed), 'wb') as f:
    pickle.dump(split_data_new, f, pickle.HIGHEST_PROTOCOL)

for split_data in split_data_new:
    train_label = data_y_array[split_data['train']]
    val_label = data_y_array[split_data['val']]
    test_label = data_y_array[split_data['test']]
    print(Counter(train_label))
    print(Counter(val_label))
    print(Counter(test_label))