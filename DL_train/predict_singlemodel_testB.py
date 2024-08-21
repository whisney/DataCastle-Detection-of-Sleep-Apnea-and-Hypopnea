import os
import numpy as np
import torch
import argparse
import shutil
from networks import Net1, Net2, Net3, Net4, Net5, Net6, Net7, Net8, Net9, Net10, Net11, Net12, Net13, Net14, Net15
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

testdata = np.load(r'../data/测试集B/test_x_B.npy')

exec('net = {}().cuda()'.format(args.net))

net.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_ACC_val.pth')))
net.eval()

if args.norm.lower() == 'norm0':
    testdata_input = testdata
elif args.norm.lower() == 'norm1':
    testdata_input = norm1(testdata)
testdata_input = torch.from_numpy(testdata_input).float().cuda()

with torch.no_grad():
    outputs = net(testdata_input)

predictions = torch.argmax(outputs, dim=1, keepdim=False).cpu().numpy().tolist()

id = [i for i in range(len(predictions))]
df = pd.DataFrame({'id': id, 'label': predictions})
df.to_csv(os.path.join(args.model_dir, 'submit_B.csv'), index=False)