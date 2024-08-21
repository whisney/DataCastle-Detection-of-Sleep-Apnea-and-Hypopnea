import pywt
import scipy
import random
import argparse
import einops
import numpy as np
from scipy import signal
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# A=100, f=50, fs=fs, phi=0, t=0.08
def sin_wave(A=100, f=50, fs=500, phi=0, t=0.4):
    '''
    :params A:    振幅
    :params f:    信号频率
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    y = einops.repeat(y, 'b -> m b', m=12)
    return y

def wgn(x, snr):
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return x + noise

# 随机选取两秒以上的信号并将其调整至15s
class ECGCropResize(object):
    def __init__(self, min_len=1000, default_len=7500):
        self.min_len = min_len
        self.default_len = default_len

    def __call__(self, data):
        crop_len = random.randint(self.min_len, self.default_len)
        crop_start = random.randint(0, self.default_len - crop_len)
        data_crop = data[:, crop_start:crop_start + crop_len]
        data_resize = np.empty_like(data)
        x = np.linspace(0, crop_len-1, crop_len)
        xnew = np.linspace(0, crop_len-1, self.default_len)

        for i in range(data.shape[0]):
            f = scipy.interpolate.interp1d(x, data_crop[i], kind='cubic')
            data_resize[i] = f(xnew)

        return data_resize

# 随机选取既定几率的一段信号并将其置零
class ECGCutOut(object):
    def __init__(self, cut_rate=0.5, default_len=7500):
        self.cut_rate = cut_rate
        self.default_len = default_len
        self.crop_len = int(self.cut_rate * self.default_len)

    def __call__(self, data):
        crop_len = random.randint(0, self.crop_len)
        crop_start = random.randint(0, self.default_len - crop_len)
        data_cutout = data.copy()
        data_cutout[:, crop_start: crop_start + crop_len] = 0
        return data_cutout

# 随机选取既定比率的频率成分并将其置零
class ECGFrequencyDropOut(object):
    def __init__(self, dropout_rate=0.1, default_len=7500):
        self.dropout_rate = dropout_rate
        self.default_len = default_len
        self.num_zeros = int(self.dropout_rate * self.default_len)

    def __call__(self, data):
        num_zeros = random.randint(0, self.num_zeros)
        channels_zero = sorted(np.random.choice(np.arange(self.default_len), num_zeros, replace=False))
        data_dct = scipy.fft.dct(data.copy())
        data_dct[:, channels_zero] = 0
        data_idct = scipy.fft.idct(data_dct)

        return data_idct

def R_peaks_detector(ecg_data, fs):
    wavelet = pywt.Wavelet('sym3')
    coeffs = pywt.wavedec(ecg_data, wavelet, level=3)
    a3, d3, d2, d1 = coeffs
    coeff = [np.zeros([len(coeff)]) for coeff in coeffs]
    coeff[2] = d2
    endata = pywt.waverec(list(coeff), wavelet)
    endata = np.square(endata)
    endata = np.convolve(endata, np.ones(10) / 10, 'same')

    min_distance = int(0.25 * fs)
    peaks, _ = signal.find_peaks(endata, distance=min_distance)
    threshold_I1 = 0.0
    threshold_I2 = 0.0
    signal_peaks = []
    noise_peaks = []
    SPKI = 0.0
    NPKI = 0.0
    RR_missed = 0
    index = 0
    indexes = []
    missed_peaks = []

    for peak in peaks:

        if endata[peak] > threshold_I1:

            signal_peaks.append(peak)
            indexes.append(index)
            SPKI = 0.125 * endata[signal_peaks[-1]] + 0.875 * SPKI

            if RR_missed != 0:
                if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                    missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                    missed_section_peaks2 = []
                    for missed_peak in missed_section_peaks:
                        if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                            -1] - missed_peak > min_distance and endata[missed_peak] > threshold_I2:
                            missed_section_peaks2.append(missed_peak)

                    if len(missed_section_peaks2) > 0:
                        missed_peak = missed_section_peaks2[np.argmax(endata[missed_section_peaks2])]
                        missed_peaks.append(missed_peak)
                        signal_peaks.append(signal_peaks[-1])
                        signal_peaks[-2] = missed_peak

        else:
            noise_peaks.append(peak)
            NPKI = 0.125 * endata[noise_peaks[-1]] + 0.875 * NPKI

        threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
        threshold_I2 = 0.5 * threshold_I1

        if len(signal_peaks) > 8:
            RR = np.diff(signal_peaks[-9:])
            RR_ave = int(np.mean(RR))
            RR_missed = int(1.66 * RR_ave)

        index = index + 1

    signal_peaks.pop(0)

    peak = peaks[0]
    if endata[peak] > threshold_I1:
        signal_peaks.insert(0, peak)
        if RR_missed != 0:
            if signal_peaks[1] - signal_peaks[0] > RR_missed:
                missed_section_peaks = peaks[indexes[0] + 1:indexes[1]]
                missed_section_peaks2 = []
                for missed_peak in missed_section_peaks:
                    if missed_peak - signal_peaks[0] > min_distance and signal_peaks[1] - missed_peak > min_distance and \
                            endata[missed_peak] > threshold_I2:
                        missed_section_peaks2.append(missed_peak)
                if len(missed_section_peaks2) > 0:
                    missed_peak = missed_section_peaks2[np.argmax(endata[missed_section_peaks2])]
                    missed_peaks.append(missed_peak)
                    signal_peaks.append(signal_peaks[1])
                    signal_peaks[0] = missed_peak
    R_peaks = sorted(set(signal_peaks))

    for i, v in enumerate(R_peaks):
        if (v - 20) > 0 and (v + 20) < len(ecg_data):
            R_peaks[i] += np.argmax(ecg_data[(v - 20):(v + 20)]) - 20

    return R_peaks

# 每个心电周期中选取一段信号置零
class ECGCycleCutOut(object):
    def __init__(self, cut_rate=0.5, fs=500):
        self.cut_rate = cut_rate
        self.fs = fs

    def __call__(self, data):
        try:
            r_peaks = R_peaks_detector(data[0], fs=self.fs)
            if len(r_peaks) > 1:
                cycle_len = int(np.mean(np.diff(r_peaks)))
                cut_len = int(self.cut_rate * cycle_len)
                cut_start = random.randint(0, cycle_len - cut_len)
                data_ = data.copy()
                for r_idx in r_peaks:
                    data_[:, r_idx + cut_start:r_idx + cut_start + cut_len] = 0
                return data_
            else:
                return data
        except:
            return data

class ChannelMask(object):
    def __init__(self, mask_rate=0.5, default_channels=12):
        self.mask_rate = mask_rate
        self.default_channels = default_channels
        self.channels = np.arange(self.default_channels)
        self.masks = int(self.mask_rate * self.default_channels)

    def __call__(self, data):
        masks = random.randint(0, self.masks)
        channels_mask = np.random.choice(self.channels, masks, replace=False)
        data_ = data.copy()
        for channel_mask in channels_mask:
            data_[channel_mask] = 0
        return data_

class ChannelMask_F(object):
    def __init__(self, mask_rate=0.5, default_channels=12):
        self.mask_rate = mask_rate
        self.default_channels = default_channels
        self.channels = np.arange(self.default_channels)
        self.masks = int(self.mask_rate * self.default_channels)

    def __call__(self, data):
        masks = random.randint(0, self.masks)
        channels_mask = np.random.choice(self.channels, masks, replace=False)
        data_ = data.copy()
        for channel_mask in channels_mask:
            data_[channel_mask] = 0
        return data_

class FrameMask(object):
    def __init__(self, mask_rate=0.5, default_frames=36):
        self.mask_rate = mask_rate
        self.default_frames = default_frames
        self.frames = np.arange(self.default_frames)
        self.masks = int(self.mask_rate * self.default_frames)

    def __call__(self, data):
        # print('data', data.shape)
        masks = random.randint(0, self.masks)
        frames_mask = np.random.choice(self.frames, masks, replace=False)
        # print('frames_mask', frames_mask)
        for frame_mask in frames_mask:
            data[:, frame_mask] = 0
        return data

class FrameMask_F(object):
    def __init__(self, mask_rate=0.5, default_frames=36):
        self.mask_rate = mask_rate
        self.default_frames = default_frames
        self.frames = np.arange(self.default_frames)
        self.masks = int(self.mask_rate * self.default_frames)

    def __call__(self, data):

        masks = random.randint(0, self.masks)
        frames_mask = np.random.choice(self.frames, masks, replace=False)

        for frame_mask in frames_mask:
            data[:, frame_mask] = 0
        return data

class ECGSineNoise(object):
    def __init__(self, min_len=1000, default_len=7500):
        self.min_len = min_len
        self.default_len = default_len
        self.fs = 500

    def __call__(self, data):
        add_noise_len = random.randint(200, 1000)
        sine_t = add_noise_len / self.fs
        crop_start = random.randint(0, add_noise_len)
        a = 0.8 * data[0, crop_start]
        hz_50 = sin_wave(A=a, t=sine_t)
        data_cutout = data.copy()
        data_cutout[:, crop_start: crop_start + add_noise_len] = data_cutout[:, crop_start: crop_start + add_noise_len] + hz_50

        return data_cutout

class ECGSineNoise_F(object):
    def __init__(self, min_len=1000, default_len=7500):
        self.min_len = min_len
        self.default_len = default_len
        self.fs = 500

    def __call__(self, data):
        add_noise_len = random.randint(200, 1000)
        sine_t = add_noise_len / self.fs
        crop_start = random.randint(0, add_noise_len)
        a = 0.8 * data[0, crop_start]
        hz_50 = sin_wave(A=a, t=sine_t)
        data_cutout = data.copy()
        data_cutout[:, crop_start: crop_start + add_noise_len] = data_cutout[:, crop_start: crop_start + add_noise_len] + hz_50

        return data_cutout

class ECGWhiteNoise(object):
    def __init__(self, min_len=1000, default_len=7500):
        self.min_len = min_len
        self.default_len = default_len
        self.snr = 80

    def __call__(self, data):

        x_noise = wgn(data, self.snr)

        return x_noise

class AddGaussian(object):
    def __init__(self, std=1, channel=0):
        self.std = std
        self.channel = channel

    def __call__(self, data):
        data_ = data.copy()
        noise = np.random.randn(data[self.channel]) * self.std
        data_[self.channel] += noise
        return data_

class Flip(object):
    def __call__(self, data):
        return np.flip(data, axis=1)

class Shift(object):
    def __init__(self, max_shiftlen=180):
        self.max_shiftlen = max_shiftlen

    def __call__(self, data):
        shift_len = random.randint(0, self.max_shiftlen)
        return np.roll(data, shift=shift_len, axis=1)