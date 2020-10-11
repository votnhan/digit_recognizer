import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def split_train_val(source_file, container, rate=0.2):
    if not os.path.exists(container):
        os.makedirs(container)

    data_frame = pd.read_csv(source_file)
    n_samples = len(data_frame)
    n_train = int((1 - rate)*n_samples)
    train_df = data_frame.iloc[:n_train]
    val_df = data_frame.iloc[n_train:]
    ctn_path = Path(container)
    train_df.to_csv(str(ctn_path / 'train.csv'))
    val_df.to_csv(str(ctn_path / 'val.csv'))

    print('Train-Val: {}-{}'.format(1-rate, rate))
    print('Split {} up {} and {}'.format(str(ctn_path), 
                str(ctn_path / 'train.csv'), str(ctn_path / 'val.csv')))

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='MNIST split train and val set')
    args_parser.add_argument('-s', '--source_file', default='train.csv', type=str, 
                        help='source csv file for training')
    args_parser.add_argument('-d', '--container', default='data', type=str, 
                        help='directory container train.csv, val.csv')
    
    args_parser.add_argument('-r', '--split_rate', default=0.2, type=float, 
                        help='val split rate')

    args = args_parser.parse_args()
    
    split_train_val(args.source_file, args.container, args.split_rate)


