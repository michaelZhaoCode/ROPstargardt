import os
import sys

import caffe
import lmdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from sklearn.metrics import roc_curve, confusion_matrix

def evaluate_model(deploy, model_path, test_lmdb, activation, model_label):
    net = caffe.Net(deploy, model_path, caffe.TEST)
    lmdb_env = lmdb.open(test_lmdb, readonly=True)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    all_true = []
    all_probs = []
    all_locs = []

    batch_size = net.blobs['data'].data.shape[0]
    batch_data = []
    batch_labels = []
    batch_locs = []

    for key, value in lmdb_cursor:
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)

        # extract locationnum as third underscore field
        fname = os.path.basename(key.decode('utf-8'))
        base = fname.split('.', 1)[0]
        parts = base.split('_')
        loc = int(parts[2])

        img = np.frombuffer(datum.data, dtype=np.uint8)
        img = img.reshape(datum.channels, datum.height, datum.width)
        img = img.astype(np.float32) / 255.0
        img = img[:, :128, :128]

        batch_data.append(img)
        batch_labels.append(int(datum.label))
        batch_locs.append(loc)

        if len(batch_data) == batch_size:
            net.blobs['data'].data[...] = np.array(batch_data)
            out = net.forward()
            if activation == 'sigmoid':
                probs = out['sigmoid'].flatten()
            else:
                probs = out['prob'][:,1]

            all_probs.extend(probs.tolist())
            all_true.extend(batch_labels)
            all_locs.extend(batch_locs)

            batch_data.clear()
            batch_labels.clear()
            batch_locs.clear()

    lmdb_env.close()

    fpr, tpr, thr = roc_curve(all_true, all_probs)
    youden = tpr - fpr
    best_thr = thr[np.argmax(youden)]
    preds = (np.array(all_probs) >= best_thr).astype(int)

    df = pd.DataFrame({
        'true': all_true,
        'pred': preds,
        'location': all_locs
    })

    def loc_acc(group):
        total1   = np.sum(group['true'] == 1)
        correct1 = np.sum((group['true'] == 1) & (group['pred'] == 1))
        acc1     = correct1 / total1 if total1 > 0 else np.nan
        total0   = np.sum(group['true'] == 0)
        correct0 = np.sum((group['true'] == 0) & (group['pred'] == 0))
        acc0     = correct0 / total0 if total0 > 0 else np.nan
        return pd.Series({'acc1': acc1, 'acc0': acc0})

    loc_df = df.groupby('location').apply(loc_acc).reset_index()
    loc_df['model'] = model_label
    return loc_df

if __name__ == '__main__':
    if len(sys.argv) != 10:
        print('Usage: python test_caffe.py '
              '<deploy.prototxt> <test_lmdb> <activation_type> '
              '<model1.caffemodel> <label1> '
              '<model2.caffemodel> <label2> '
              '<model3.caffemodel> <label3>')
        sys.exit(1)

    deploy     = sys.argv[1]
    test_lmdb  = sys.argv[2]
    activation = sys.argv[3].lower()
    if activation not in ['softmax','sigmoid']:
        print('Error: activation_type must be softmax or sigmoid')
        sys.exit(1)

    model_paths = [sys.argv[4], sys.argv[6], sys.argv[8]]
    labels      = [sys.argv[5], sys.argv[7], sys.argv[9]]

    os.makedirs('plots', exist_ok=True)

    results_list = []
    for path, label in zip(model_paths, labels):
        df_loc = evaluate_model(deploy, path, test_lmdb, activation, label)
        results_list.append(df_loc)
    results = pd.concat(results_list, ignore_index=True)

    # Plot for class 1
    plt.figure()
    for label in labels:
        sub = results[results['model'] == label].dropna(subset=['acc1'])
        plt.plot(sub['location'], sub['acc1'], marker='o', label=label)
    plt.title('Class 1 Accuracy by Location')
    plt.xlabel('Location')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/class1_accuracy_by_location.png')
    plt.show()

    # Plot for class 0
    plt.figure()
    for label in labels:
        sub = results[results['model'] == label].dropna(subset=['acc0'])
        plt.plot(sub['location'], sub['acc0'], marker='o', label=label)
    plt.title('Class 0 Accuracy by Location')
    plt.xlabel('Location')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/class0_accuracy_by_location.png')
    plt.show()
