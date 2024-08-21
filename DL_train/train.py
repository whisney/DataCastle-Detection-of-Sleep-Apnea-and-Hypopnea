import os
from dataset import Dataset_all
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
import argparse
import random
import shutil
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from utils import poly_lr_scheduler, learning_rate_seq
from networks import Net1, Net2, Net3, Net4, Net5, Net6, Net7, Net8, Net9, Net10, Net11, Net12, Net13, Net14, Net15, \
    Net16, Net17, Net18, Net19, Net20, Net21, Net22, Net23, Net24
from loss_function.CB_Loss import CB_loss
from norms import norm1, norm2
import pandas as pd
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--name', type=str, default='base', help='name')
parser.add_argument('--pretrained', type=str, default='none', help='pretrained model path')
parser.add_argument('--epoch', type=int, default=2000, help='all_epochs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--fold', type=int, default=0, help='fold of cross validation')
parser.add_argument('--net', type=str, default='base', help='net')
parser.add_argument('--norm', type=str, default='norm1', help='norm')
parser.add_argument('--resample', type=str, default='resample0', help='resample')
parser.add_argument('--aug', type=str, default='aug0', help='aug')
parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam/adamw')
parser.add_argument('--loss', type=str, default='cbloss', help='cbloss/celoss')
parser.add_argument('--lr_strategy', type=str, default='step', help='step/poly')
parser.add_argument('--split', type=str, default='train_val_test_4folds_20230726', help='split pkl name')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_path = r'../data/训练集/train_x.npy'
label_path = r'../data/训练集/train_y.npy'
split_file = r'../data/{}.pkl'.format(args.split)

loss_threshold = 0.0
# n_bs_per_step = 16 // args.bs + 1
num_class = 3

save_dir = 'trained_models/{}/{}_{}_{}_{}_{}_{}_{}_{}/bs{}_epoch{}_seed{}_fold{}'.format(
    args.split, args.name, args.resample, args.norm, args.net, args.aug, args.lr_strategy, args.loss, args.optimizer, args.bs, args.epoch, args.seed, args.fold)

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
test_writer = SummaryWriter(os.path.join(save_dir, 'log/test'), flush_secs=2)
print(save_dir)

print('dataset loading')

train_data = Dataset_all(path_x=data_path, path_y=label_path, path_split=split_file, fold=args.fold, set='train',
                         norm_mode=args.norm, aug_mode=args.aug, aug=True, resample=args.resample, seed=args.seed)
val_data = Dataset_all(path_x=data_path, path_y=label_path, path_split=split_file, fold=args.fold, set='val',
                         norm_mode=args.norm, aug_mode=args.aug, aug=False, resample='resample0', seed=args.seed)
test_data = Dataset_all(path_x=data_path, path_y=label_path, path_split=split_file, fold=args.fold, set='test',
                         norm_mode=args.norm, aug_mode=args.aug, aug=False, resample='resample0', seed=args.seed)

train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

print('train_lenth: %i  val_lenth: %i  test_lenth: %i' % (
    train_data.len, val_data.len, test_data.len))

exec('net = {}().cuda()'.format(args.net))

if not args.pretrained == 'none':
    net.load_state_dict(torch.load(args.pretrained))

if args.optimizer.lower() == 'adamw1':
    lr_max, weight_decay = 0.00001, 0.0005
    optimizer = optim.AdamW(net.parameters(), lr=lr_max, weight_decay=weight_decay)
elif args.optimizer.lower() == 'adamw2':
    lr_max, weight_decay = 0.000005, 0.0001
    optimizer = optim.AdamW(net.parameters(), lr=lr_max, weight_decay=weight_decay)
elif args.optimizer.lower() == 'adam1':
    lr_max, weight_decay = 0.00001, 0.0005
    optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=weight_decay)
elif args.optimizer.lower() == 'sgd1':
    lr_max, weight_decay, momentum = 0.005, 5e-3, 0.9
    optimizer = optim.SGD(net.parameters(), momentum=momentum, lr=lr_max, weight_decay=weight_decay)
elif args.optimizer.lower() == 'sgd2':
    lr_max, momentum = 0.001, 0.99
    optimizer = optim.SGD(net.parameters(), momentum=momentum, lr=lr_max)
elif args.optimizer.lower() == 'radam1':
    lr_max = 0.001
    optimizer = optim.RAdam(net.parameters(), lr=lr_max)

if args.loss.lower() == 'celoss':
    loss_fuc = torch.nn.CrossEntropyLoss()
elif args.loss.lower() == 'cbloss1':
    loss_fuc = partial(CB_loss, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2],
                           no_of_classes=num_class, loss_type='focal', beta=0.999, gamma=2)
elif args.loss.lower() == 'cbloss2':
    loss_fuc = partial(CB_loss, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2],
                           no_of_classes=num_class, loss_type='softmax', beta=0.999, gamma=2)
elif args.loss.lower() == 'cbloss3':
    loss_fuc = partial(CB_loss, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2],
                           no_of_classes=num_class, loss_type='focal', beta=0.99999, gamma=2)
elif args.loss.lower() == 'cbloss4':
    loss_fuc = partial(CB_loss, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2],
                           no_of_classes=num_class, loss_type='focal', beta=0.999999, gamma=2)
elif args.loss.lower() == 'cbloss5':
    loss_fuc = partial(CB_loss, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2 / 2],
                           no_of_classes=num_class, loss_type='focal', beta=0.999999, gamma=2)
elif args.loss.lower() == 'cbloss6':
    loss_fuc = partial(CB_loss, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2 // 10 * 8],
                           no_of_classes=num_class, loss_type='focal', beta=0.999999, gamma=2)
elif args.loss.lower() == 'cbloss7':
    loss_fuc = partial(CB_loss, samples_per_cls=[train_data.num_0, train_data.num_1, train_data.num_2 // 10 * 6],
                           no_of_classes=num_class, loss_type='focal', beta=0.999999, gamma=2)
elif args.loss.lower() == 'wceloss1':
    loss_fuc = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 6, 9]).cuda())
elif args.loss.lower() == 'wceloss2':
    loss_fuc = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 0.4, 0.4]).cuda())
elif args.loss.lower() == 'wceloss3':
    loss_fuc = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 12, 9]).cuda())
elif args.loss.lower() == 'wceloss4':
    loss_fuc = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 10, 10]).cuda())

lr_scheduler = MultiStepLR(optimizer, milestones=[int((6 / 10) * args.epoch), int((9 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)
lrs_seq = learning_rate_seq(num_epochs=args.epoch, learning_rate=lr_max)

best_ACC_val = 0
test_ACC_when_best_ACC_val = 0

print('training')

for epoch in range(args.epoch):
    if args.lr_strategy.lower() == 'poly':
        poly_lr_scheduler(optimizer, lr_max, epoch, lr_decay_iter=1, max_iter=args.epoch, power=0.9)
    elif args.lr_strategy.lower() == 'seq':
        for param_gourp in optimizer.param_groups:
            param_gourp['lr'] = lrs_seq[epoch]

    lr = optimizer.param_groups[0]['lr']
    net.train()
    train_epoch_loss = []
    train_epoch_class_label = []
    train_epoch_pred_class = []
    for i, (imgs, labels) in enumerate(train_dataloader):
        imgs = imgs.cuda().float()
        labels = labels.cuda().long()
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_fuc(outputs, labels)
        loss = (loss - loss_threshold).abs() + loss_threshold
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        outputs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
        train_epoch_loss.append(loss.item())
        train_epoch_class_label.append(labels.cpu().numpy())
        train_epoch_pred_class.append(predicted.cpu().numpy())
        # print('[%d/%d, %d/%d] train_loss: %.3f' %
        #       (epoch + 1, args.epoch, i + 1, len(train_dataloader), loss.item()))
    if args.lr_strategy.lower() == 'step':
        lr_scheduler.step()

    with torch.no_grad():
        net.eval()
        val_epoch_loss = []
        val_epoch_class_label = []
        val_epoch_pred_class = []
        for i, (imgs, labels) in enumerate(val_dataloader):
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()
            outputs = net(imgs)
            loss = loss_fuc(outputs, labels)
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
            val_epoch_loss.append(loss.item())
            val_epoch_class_label.append(labels.cpu().numpy())
            val_epoch_pred_class.append(predicted.cpu().numpy())

    with torch.no_grad():
        net.eval()
        test_epoch_loss = []
        test_epoch_class_label = []
        test_epoch_pred_class = []
        for i, (imgs, labels) in enumerate(test_dataloader):
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()
            outputs = net(imgs)
            loss = loss_fuc(outputs, labels)
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
            test_epoch_loss.append(loss.item())
            test_epoch_class_label.append(labels.cpu().numpy())
            test_epoch_pred_class.append(predicted.cpu().numpy())

    train_epoch_class_label = np.concatenate(train_epoch_class_label)
    train_epoch_pred_class = np.concatenate(train_epoch_pred_class)
    val_epoch_class_label = np.concatenate(val_epoch_class_label)
    val_epoch_pred_class = np.concatenate(val_epoch_pred_class)
    test_epoch_class_label = np.concatenate(test_epoch_class_label)
    test_epoch_pred_class = np.concatenate(test_epoch_pred_class)

    train_ACC = accuracy_score(train_epoch_class_label, train_epoch_pred_class)
    val_ACC = accuracy_score(val_epoch_class_label, val_epoch_pred_class)
    test_ACC = accuracy_score(test_epoch_class_label, test_epoch_pred_class)

    train_epoch_loss = np.mean(train_epoch_loss)
    val_epoch_loss = np.mean(val_epoch_loss)
    test_epoch_loss = np.mean(test_epoch_loss)

    print(
        '[%d/%d] train_loss: %.3f train_ACC: %.3f val_ACC: %.3f test_ACC: %.3f' %
        (epoch, args.epoch, train_epoch_loss, train_ACC, val_ACC, test_ACC))

    if val_ACC > best_ACC_val:
        best_ACC_val = val_ACC
        test_ACC_when_best_ACC_val = test_ACC
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_ACC_val.pth'))

    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('loss', train_epoch_loss, epoch)
    train_writer.add_scalar('ACC', train_ACC, epoch)

    val_writer.add_scalar('loss', val_epoch_loss, epoch)
    val_writer.add_scalar('ACC', val_ACC, epoch)
    val_writer.add_scalar('best_ACC_val', best_ACC_val, epoch)

    test_writer.add_scalar('loss', test_epoch_loss, epoch)
    test_writer.add_scalar('ACC', test_ACC, epoch)

    if (epoch + 1) == args.epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)
print('Val ACC: {}'.format(best_ACC_val))
print('Test ACC: {}'.format(test_ACC_when_best_ACC_val))

net.load_state_dict(torch.load(os.path.join(save_dir, 'best_ACC_val.pth')))
net.eval()

testdata = np.load(r'../data/测试集A/test_x_A.npy')
if args.norm.lower() == 'norm0':
    pass
elif args.norm.lower() == 'norm1':
    testdata = norm1(testdata)
elif args.norm.lower() == 'norm2':
    testdata = norm2(testdata)
testdata = torch.from_numpy(testdata).float().cuda()
with torch.no_grad():
    outputs = net(testdata)
outputs = torch.softmax(outputs, dim=1)
predicted = torch.argmax(outputs, dim=1, keepdim=False).cpu().numpy().tolist()
id = [i for i in range(len(predicted))]
df = pd.DataFrame({'id': id, 'label': predicted})
df.to_csv(os.path.join(save_dir, 'submit_A.csv'), index=False)