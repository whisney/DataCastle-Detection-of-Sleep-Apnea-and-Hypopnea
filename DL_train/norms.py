import numpy as np

def norm1(data):
    data_ = np.copy(data).astype(np.float32)
    signal1_clip = (90, 99)
    signal2_clip = (50, 80)
    if len(data_.shape) == 2:
        data_[0] = (np.clip(data_[0], signal1_clip[0], signal1_clip[1]) - signal1_clip[0]) / (signal1_clip[1] - signal1_clip[0])
        data_[1] = (np.clip(data_[1], signal2_clip[0], signal2_clip[1]) - signal2_clip[0]) / (signal2_clip[1] - signal2_clip[0])
    elif len(data_.shape) == 3:
        data_[:, 0] = (np.clip(data_[:, 0], signal1_clip[0], signal1_clip[1]) - signal1_clip[0]) / (signal1_clip[1] - signal1_clip[0])
        data_[:, 1] = (np.clip(data_[:, 1], signal2_clip[0], signal2_clip[1]) - signal2_clip[0]) / (signal2_clip[1] - signal2_clip[0])
    return data_

def norm2(data):
    data_ = np.copy(data).astype(np.float32)
    signal1_mean, signal1_std = 94.68028191311501, 3.0147888371863765
    signal2_mean, signal2_std = 66.17522896008475, 10.509252393854648
    if len(data_.shape) == 2:
        data_[0] = (data_[0] - signal1_mean) / signal1_std
        data_[1] = (data_[1] - signal2_mean) / signal2_std
    elif len(data_.shape) == 3:
        data_[:, 0] = (data_[:, 0] - signal1_mean) / signal1_std
        data_[:, 1] = (data_[:, 1] - signal2_mean) / signal2_std
    return data_