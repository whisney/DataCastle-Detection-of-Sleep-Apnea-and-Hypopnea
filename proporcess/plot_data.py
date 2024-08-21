import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

data_x = np.load(r'E:\competition\睡眠呼吸暂停和低通气事件检测\data\训练集/train_x.npy')
data_y = np.load(r'E:\competition\睡眠呼吸暂停和低通气事件检测\data\训练集/train_y.npy')

num = data_x.shape[0] // 5

x = [_ for _ in range(180)]
for i in range(num):
    print('{} ~ {}'.format(5 * i, 5 * i + 5))
    print(data_y[5 * i: 5 * i + 5])
    plt.figure(figsize=(15, 6))
    plt.subplot(251)
    plt.plot(x, data_x[5 * i + 0, 0])
    plt.ylim(90, 99)
    plt.subplot(252)
    plt.plot(x, data_x[5 * i + 1, 0])
    plt.ylim(90, 99)
    plt.subplot(253)
    plt.plot(x, data_x[5 * i + 2, 0])
    plt.ylim(90, 99)
    plt.subplot(254)
    plt.plot(x, data_x[5 * i + 3, 0])
    plt.ylim(90, 99)
    plt.subplot(255)
    plt.plot(x, data_x[5 * i + 4, 0])
    plt.ylim(90, 99)

    plt.subplot(256)
    plt.plot(x, data_x[5 * i + 0, 1])
    plt.ylim(55, 75)
    plt.subplot(257)
    plt.plot(x, data_x[5 * i + 1, 1])
    plt.ylim(55, 75)
    plt.subplot(258)
    plt.plot(x, data_x[5 * i + 2, 1])
    plt.ylim(55, 75)
    plt.subplot(259)
    plt.plot(x, data_x[5 * i + 3, 1])
    plt.ylim(55, 75)
    plt.subplot(2,5,10)
    plt.plot(x, data_x[5 * i + 4, 1])
    plt.ylim(55, 75)
    plt.show()