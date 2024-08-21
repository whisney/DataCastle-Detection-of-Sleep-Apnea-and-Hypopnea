import torch
import numpy as np
from torch.utils.data import Dataset
from norms import norm1, norm2
import pickle
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
import torchvision.transforms as transforms
from Augment import FrameMask, ChannelMask, Shift, Flip, ECGCropResize,  AddGaussian, ECGCutOut
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids, CondensedNearestNeighbour, InstanceHardnessThreshold, NearMiss, \
    RandomUnderSampler

class Dataset_all(Dataset):
    def __init__(self, path_x, path_y, path_split, fold, set, norm_mode='norm1', aug_mode='aug0', aug=True,
                 resample='resample0', seed=0):
        super(Dataset_all, self).__init__()
        self.norm_mode = norm_mode
        self.aug_mode = aug_mode
        self.aug = aug
        split_data = pickle.load(open(path_split, 'rb'))

        if fold == 5:
            if set == 'train':
                self.data_index = split_data[0]['train'] + split_data[0]['val'] + split_data[0]['test']
            else:
                self.data_index = split_data[0][set]
        elif fold > 5:
            if set == 'train':
                self.data_index = split_data[fold - 6]['train'] + split_data[fold - 6]['val']
            else:
                self.data_index = split_data[fold - 6]['test']
        else:
            self.data_index = split_data[fold][set]
        self.data = np.load(path_x)[self.data_index]
        self.label = np.load(path_y)[self.data_index]

        if resample.lower() == 'resample0':
            pass
        elif resample.lower() == 'resample11':
            zero_index = list(np.where(self.label == 0)[0])
            np.random.shuffle(zero_index)
            total_index = zero_index[:4000] + list(np.where(self.label != 0)[0])
            self.data = self.data[total_index]
            self.label = self.label[total_index]
        else:
            n_samples, n_channels, n_features = self.data.shape
            X_reshaped = self.data.reshape(len(self.data), -1)
            if resample.lower() == 'resample1':
                Sampler = RandomOverSampler(random_state=seed)
            elif resample.lower() == 'resample2':
                Sampler = ClusterCentroids(estimator=MiniBatchKMeans(n_init=1, random_state=seed), random_state=seed)
            elif resample.lower() == 'resample3':
                Sampler = CondensedNearestNeighbour(random_state=seed, n_jobs=-1)
            elif resample.lower() == 'resample4':
                Sampler = InstanceHardnessThreshold(sampling_strategy='majority', random_state=seed, n_jobs=-1)
            elif resample.lower() == 'resample5':
                Sampler = NearMiss(n_jobs=-1)
            elif resample.lower() == 'resample6':
                Sampler = RandomUnderSampler(random_state=seed)
            elif resample.lower() == 'resample7':
                Sampler = SMOTE(random_state=seed, n_jobs=-1)
            elif resample.lower() == 'resample8':
                Sampler = ADASYN(random_state=seed, n_jobs=-1)
            elif resample.lower() == 'resample9':
                Sampler = BorderlineSMOTE(random_state=seed, n_jobs=-1)
            elif resample.lower() == 'resample10':
                Sampler = KMeansSMOTE(kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=seed), random_state=seed, n_jobs=-1)
            self.data, self.label = Sampler.fit_resample(X_reshaped, self.label)
            self.data = self.data.reshape(-1, n_channels, n_features)

        self.len = len(self.data)
        self.num_0 = np.sum(self.label == 0)
        self.num_1 = np.sum(self.label == 1)
        self.num_2 = np.sum(self.label == 2)

        if aug_mode.lower() == 'aug1':
            self.my_transforms = transforms.Compose([
                transforms.RandomApply([FrameMask(mask_rate=0.4)]),
                transforms.RandomApply([ChannelMask(mask_rate=0.4, default_channels=2)]),
                transforms.RandomApply([ECGCutOut(default_len=180)]),
                transforms.RandomApply([ECGCropResize(min_len=30, default_len=180)]),
            ])
        elif aug_mode.lower() == 'aug2':
            self.my_transforms = transforms.Compose([
                transforms.RandomApply([ECGCropResize(min_len=60, default_len=180)], p=0.2),
                transforms.RandomApply([FrameMask(mask_rate=0.4, default_frames=180)], p=0.2),
                transforms.RandomApply([ChannelMask(mask_rate=0.5, default_channels=2)], p=0.2),
                transforms.RandomApply([ECGCutOut(cut_rate=0.5, default_len=180)], p=0.3),
                transforms.RandomApply([Flip()], p=0.2),
                transforms.RandomApply([Shift(max_shiftlen=180)], p=0.2),
                transforms.RandomApply([AddGaussian(std=0.1, channel=0)], p=0.2),
                transforms.RandomApply([AddGaussian(std=0.5, channel=1)], p=0.2),
            ])
        elif aug_mode.lower() == 'aug3':
            self.my_transforms = transforms.Compose([
                transforms.RandomApply([ECGCropResize(min_len=60, default_len=180)], p=0.5),
                transforms.RandomApply([FrameMask(mask_rate=0.4, default_frames=180)], p=0.5),
                transforms.RandomApply([ChannelMask(mask_rate=0.5, default_channels=2)], p=0.5),
                transforms.RandomApply([ECGCutOut(cut_rate=0.5, default_len=180)], p=0.5),
                transforms.RandomApply([Flip()], p=0.5),
                transforms.RandomApply([Shift(max_shiftlen=30)], p=0.5),
            ])
        elif aug_mode.lower() == 'aug4':
            self.my_transforms = transforms.Compose([
                transforms.RandomApply([ECGCropResize(min_len=60, default_len=180)], p=0.1),
                transforms.RandomApply([FrameMask(mask_rate=0.4, default_frames=180)], p=0.1),
                transforms.RandomApply([ChannelMask(mask_rate=0.5, default_channels=2)], p=0.2),
                transforms.RandomApply([ECGCutOut(cut_rate=0.5, default_len=180)], p=0.2),
            ])

    def __getitem__(self, index):
        data_one = self.data[index]
        label_one = self.label[index]

        if self.norm_mode.lower() == 'norm0':
            pass
        elif self.norm_mode.lower() == 'norm1':
            data_one = norm1(data_one)
        elif self.norm_mode.lower() == 'norm2':
            data_one = norm2(data_one)

        if self.aug:
            if self.aug_mode.lower() == 'aug0':
                pass
            elif self.aug_mode.lower() in ['aug1']:
                data_one = self.my_transforms(data_one)

        data_one = torch.from_numpy(data_one).float()
        label_one = torch.tensor(int(label_one))
        return data_one, label_one

    def __len__(self):
        return self.len