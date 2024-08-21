import os
import pandas as pd
import numpy as np

def calculate_proportion(lst):
    total_count = len(lst)
    count_0 = lst.count(0)
    count_1 = lst.count(1)
    count_2 = lst.count(2)
    return (count_0 / total_count, count_1 / total_count, count_2 / total_count)

def distance(proportion1, proportion2):
    return sum(abs(a - b) for a, b in zip(proportion1, proportion2))

def sort_lists_by_proportion(A, named_lists):
    A_proportion = calculate_proportion(A)
    list_proportions = [(name, lst, calculate_proportion(lst)) for name, lst in named_lists]
    sorted_named_lists = sorted(list_proportions, key=lambda x: distance(A_proportion, x[2]))
    return [name for name, lst, _ in sorted_named_lists]

if __name__ == '__main__':
    base_list = list(pd.read_csv(r'/home/zyw/competitions/DetectionSAH/DL_train/pred_testB/ensemble19/submit_B.csv')['label'])
    named_N_lists = []
    data_dir = r'/home/zyw/competitions/DetectionSAH/DL_train/trained_models'
    for split in os.listdir(data_dir):
        for pset in os.listdir(os.path.join(data_dir, split)):
            for bset in os.listdir(os.path.join(data_dir, split, pset)):
                if os.path.exists(os.path.join(data_dir, split, pset, bset, 'submit_B.csv')):
                    named_N_lists.append((os.path.join(data_dir, split, pset, bset),
                                          list(pd.read_csv(os.path.join(data_dir, split, pset, bset, 'submit_B.csv'))['label'])))

    sorted_named_N_lists = sort_lists_by_proportion(base_list, named_N_lists)
    for name in sorted_named_N_lists:
        print(name)