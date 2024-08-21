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
parser.add_argument('--net', nargs='*', type=str, default='base', help='net')
parser.add_argument('--norm', nargs='*', type=str, default='norm1', help='norm')
parser.add_argument('--model_dir', nargs='*', type=str, default='norm1', help='norm')
parser.add_argument('--id', type=int, default=1, help='id')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

save_dir = 'pred_testA/ensemble{}'.format(args.id)
os.makedirs(save_dir, exist_ok=True)

testdata = np.load(r'../data/测试集A/test_x_A.npy')

predictions = 0
for i, net_name in enumerate(args.net):
    exec('net = {}().cuda()'.format(net_name))

    net.load_state_dict(torch.load(os.path.join(args.model_dir[i], 'best_ACC_val.pth')))
    net.eval()

    if args.norm[i].lower() == 'norm0':
        testdata_input = testdata
    elif args.norm[i].lower() == 'norm1':
        testdata_input = norm1(testdata)
    testdata_input = torch.from_numpy(testdata_input).float().cuda()

    with torch.no_grad():
        outputs = net(testdata_input)
    predictions += torch.softmax(outputs, dim=1)

predictions = torch.argmax(predictions, dim=1, keepdim=False).cpu().numpy().tolist()

id = [i for i in range(len(predictions))]
df = pd.DataFrame({'id': id, 'label': predictions})
df.to_csv(os.path.join(save_dir, 'submit_A.csv'), index=False)