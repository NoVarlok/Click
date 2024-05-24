import os
import numpy as np
import pandas as pd
import json
import multiprocessing as mp
import argparse
import csv
from tabulate import tabulate


def is_negative(x):
    return x[0] > 0.5


def process(infer_data_path):
    #print('\t', infer_data_path)
    if not os.path.exists(f'{infer_data_path}/gen.txt'):
        return []

    stats_line = []

    first_gen = json.loads(open(f'{infer_data_path}/gen.txt').readline())['generation']
    if isinstance(first_gen, str):
        multiple = 1
    else:
        multiple = len(first_gen)

    if os.path.exists(f'{infer_data_path}/pred_list.txt'):
        preds = [json.loads(e) for e in open(f'{infer_data_path}/pred_list.txt')]
        preds = np.array(preds).reshape(-1, multiple, 2).transpose(2, 0, 1)
        emp_prob = (is_negative(preds).sum(axis=-1) > 0).mean()
        stats_line.extend([str(emp_prob)])
    else:
        stats_line.extend([''])

    if os.path.exists(f'{infer_data_path}/loss_large_list.txt'):
        losses = [json.loads(e) for e in open(f'{infer_data_path}/loss_large_list.txt')]
        ppls = [np.exp(e['loss'] / e['num_tokens']) for e in losses]
        ppl = np.mean(ppls)
        stats_line.append(ppl)
    else:
        stats_line.extend(['', ])

    if os.path.exists(f'{infer_data_path}/dist_list.txt'):
        dists = [json.loads(e) for e in open(f'{infer_data_path}/dist_list.txt')]
        dists = np.array(dists).mean(0)
        dists = [str(e) for e in dists][-2:]
        stats_line.extend(dists)
    else:
        stats_line.extend(['', '', ])

    return stats_line


def main(args):
    if not args.test:
        save_name = './results_bad.txt'
    else:
        save_name = './results_bad_test.txt'
    df_dict = {
        'Method': [],
    }
    for column in args.columns:
        df_dict[column] = []
    for path, name in zip(args.infer_data_paths, args.names):
        stats = process(path)
        df_dict['Method'].append(name)
        for column, value in zip(args.columns, stats):
            df_dict[column].append(value)
    df = pd.DataFrame.from_dict(df_dict)
    # df.to_csv(save_name, index=False)
    with open(save_name, 'w') as fout:
        # print(tabulate(df, headers='keys', tablefmt='pipe', stralign='left', showindex=False), file=fout)
        print(df.to_markdown(index=False,), file=fout)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_data_paths', type=str, nargs='+')
    parser.add_argument('--names', type=str, nargs='+')
    parser.add_argument('--columns', type=str, nargs='+')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    print('infer_data_paths:', len(args.infer_data_paths))
    print('names:', len(args.names))

    main(args)
