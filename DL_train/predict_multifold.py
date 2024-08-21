import os
import numpy as np
import torch
import argparse
import shutil
from networks import Net1, Net2, Net3, Net4, Net5, Net6, Net7, Net8, Net9, Net10, Net11, Net12, Net13, Net14, Net15, \
    Net16, Net17, Net18, Net19, Net20, Net22, Net23
from norms import norm1
import pandas as pd
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--net', type=str, default='base', help='net')
parser.add_argument('--norm', type=str, default='norm1', help='norm')
parser.add_argument('--model_dir', type=str, default='norm1', help='norm')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

model_dir_split = args.model_dir.split('/')
save_dir = 'pred_testA/{}/{}/{}'.format(model_dir_split[-3], model_dir_split[-2], model_dir_split[-1])
os.makedirs(save_dir, exist_ok=True)

exec('net = {}().cuda()'.format(args.net))

testdata = np.load(r'../data/测试集A/test_x_A.npy')
if args.norm.lower() == 'norm0':
    pass
if args.norm.lower() == 'norm1':
    testdata = norm1(testdata)
testdata = torch.from_numpy(testdata).float().cuda()
predictions = 0

dir_list = glob.glob('{}_fold*'.format(args.model_dir))
for dir_one in dir_list:
    print(dir_one)
    net.load_state_dict(torch.load(os.path.join(dir_one, 'best_ACC_val.pth')))
    net.eval()
    with torch.no_grad():
        outputs = net(testdata)
    predictions += torch.softmax(outputs, dim=1)

predictions = torch.argmax(predictions, dim=1, keepdim=False).cpu().numpy().tolist()

id = [i for i in range(len(predictions))]
df = pd.DataFrame({'id': id, 'label': predictions})
df.to_csv(os.path.join(save_dir, 'submit_A.csv'), index=False)