import numpy as np

def poly_lr_scheduler(optimizer, base_lr, n_iter, lr_decay_iter=1, max_iter=100, power=0.9):
    if n_iter % lr_decay_iter == 0 and n_iter <= max_iter:
        lr = base_lr * (1 - n_iter / max_iter) ** power
        for param_gourp in optimizer.param_groups:
            param_gourp['lr'] = lr

def learning_rate_seq(num_epochs=100, learning_rate=0.01):
    one_cycle = num_epochs
    half_len = int(one_cycle * 0.45)
    x1 = np.linspace(0.1 * learning_rate, learning_rate, half_len)
    x2 = np.linspace(x1[-1], 0.1 * learning_rate, half_len + 1)[1:]
    x3 = np.linspace(x2[-1], 0.001 * learning_rate, one_cycle - 2 * half_len + 1)[1:]
    x = np.concatenate([x1, x2, x3])
    return x