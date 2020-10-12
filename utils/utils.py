import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

def read_json(fname):
    with open(fname, 'r') as infile:
        data = json.load(infile)
    
    return data

def save_pandas_df(data, filename, index, columns, use_index=True):
    df = pd.DataFrame(data=data, index=index, columns=columns)
    df.to_csv(filename, index=use_index)

def create_folder(path):
    path = str(path)
    if not os.path.exists(path):
        os.makedirs(path)

def append_log_to_file(file_path, list_items):
    with open(file_path, 'a') as opened_file:
        line_items = ','.join(list_items)
        opened_file.write(line_items+'\n')
        opened_file.close()

def plot_train_val_loss(log_file, out_file):
    df = pd.read_csv(log_file, index_col='Epoch')
    plt.plot(df['Train_loss'].values, label='Training loss')
    plt.plot(df['Validation_loss'].values, label='Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(out_file)
    print('Plot train and val loss to {}'.format(out_file))

