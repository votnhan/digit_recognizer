import pandas as pd
import numpy as np
import os

n_classes = 10

def precision_recall(prediction_file, output_dir):
    df = pd.read_csv(prediction_file, index_col='Unnamed: 0')
    target = np.array(df['Target'].values)
    prediction = np.array(df['Prediction'].values)
    precision, recall, f1_list = [], [], []
    for i in range(n_classes):
        tg_for_class = target == i
        pred_for_class = prediction[tg_for_class]
        tp = np.sum(pred_for_class == i)
        prec = tp / np.sum(prediction == i)
        rec = tp / np.sum(tg_for_class)
        f1 = 2 * (prec*rec) / (prec + rec)
        precision.append(prec)
        recall.append(rec)
        f1_list.append(f1)

    indexs = [i for i in range(n_classes)]
    cls_metrics_df = pd.DataFrame(zip(precision, recall, f1_list), 
                        index=indexs, columns=['Precision', 'Recall', 'F1'])
    pred_filename, _ = os.path.basename(prediction_file).split('.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, '{}_stat.csv'.format(pred_filename))
    cls_metrics_df.to_csv(output_file)

