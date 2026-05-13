#!/usr/bin/env python3
import sys
import os
import lmdb
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from scipy import stats
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def evaluate_model(deploy_prototxt, caffe_model, test_lmdb, activation_type):
    # Runs the given Caffe model on test_lmdb and returns:
    #  - true_labels: ndarray of shape (N,)
    #  - probs:       ndarray of shape (N,), probability of the positive class
    caffe.set_mode_gpu()
    caffe.set_devices([0,1])
    net = caffe.Net(deploy_prototxt, caffe_model, caffe.TEST)

    env = lmdb.open(test_lmdb, readonly=True)
    txn = env.begin()
    cursor = txn.cursor()

    batch_size = net.blobs['data'].data.shape[0]
    batch_data = []
    batch_labels = []
    true_labels = []
    probs = []

    for _, value in cursor:
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        img = np.frombuffer(datum.data, dtype=np.uint8)
        img = img.reshape(datum.channels, datum.height, datum.width)
        img = img.astype(np.float32) / 255.0
        img = img[:, :128, :128]

        batch_data.append(img)
        batch_labels.append(int(datum.label))

        if len(batch_data) == batch_size:
            net.blobs['data'].data[...] = np.array(batch_data)
            out = net.forward()

            if activation_type == "sigmoid":
                p = out['sigmoid'].flatten()
            else:
                p = out['prob'][:, 1]

            true_labels.extend(batch_labels)
            probs.extend(p.tolist())

            batch_data.clear()
            batch_labels.clear()

    env.close()
    return np.array(true_labels), np.array(probs)


def tost_equivalence_test(diffs, delta, alpha):
    # Performs Two One-Sided Tests (TOST) for equivalence
    # H01: mean(diffs) <= -delta vs HA: mean(diffs) > -delta
    # H02: mean(diffs) >=  delta vs HA: mean(diffs) <  delta
    n = len(diffs)
    mean_d = np.mean(diffs)
    sd_d = np.std(diffs, ddof=1)
    se = sd_d / np.sqrt(n)
    df = n - 1

    # Test 1: H0: mean_d <= -delta vs HA: mean_d > -delta
    t1 = (mean_d + delta) / se
    p1 = stats.t.sf(t1, df)

    # Test 2: H0: mean_d >= delta vs HA: mean_d < delta
    t2 = (mean_d - delta) / se
    p2 = stats.t.cdf(t2, df)

    equivalent = (p1 < alpha) and (p2 < alpha)
    return t1, p1, t2, p2, equivalent


def plot_tost_ci(mean_d, se, n, delta, alpha=0.05, out_path="plots/tost_ci.png"):
    """
    Plot the (1-2*alpha)% CI around the mean difference,
    with vertical lines at -delta, 0, and +delta.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # For TOST at level alpha, use a (1 - 2*alpha) CI:
    # e.g. alpha=0.05 -> 90% CI
    ci_level = 1 - 2*alpha
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df)   # one-sided -> same for two-sided 1-2α
    half_width = t_crit * se

    ci_lo = mean_d - half_width
    ci_hi = mean_d + half_width

    plt.figure(figsize=(6,3))
    # plot the CI as a horizontal line
    plt.hlines(1, ci_lo, ci_hi, linewidth=4)
    # plot the point estimate
    plt.plot(mean_d, 1, 'o')
    # plot the equivalence bounds
    plt.vlines([-delta, 0, delta], 0.9, 1.1, linestyles=['--','-','--'], 
               colors=['gray','black','gray'], label='bounds')
    plt.yticks([])  # hide y-axis
    plt.xlabel('Mean difference (model1 - model2)')
    plt.title(f'{int(ci_level*100)}% CI vs equivalence bounds')
    plt.legend(['Mean','Equivalence bounds'], loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"TOST CI plot saved to {out_path}")


def main():
    if len(sys.argv) not in (7, 8):
        print("Usage: python equivalence_test.py \
              <deploy.prototxt> <model1.caffemodel> <model2.caffemodel> \
              <test_lmdb> <activation_type> <delta> [alpha]")
        sys.exit(1)

    deploy = sys.argv[1]
    model1 = sys.argv[2]
    model2 = sys.argv[3]
    lmdb_path = sys.argv[4]
    activation_type = sys.argv[5]
    delta = float(sys.argv[6])
    alpha = float(sys.argv[7]) if len(sys.argv) == 8 else 0.05

    print("Evaluating Model 1...")
    true1, prob1 = evaluate_model(deploy, model1, lmdb_path, activation_type)
    print("Evaluating Model 2...")
    true2, prob2 = evaluate_model(deploy, model2, lmdb_path, activation_type)

    if not np.array_equal(true1, true2):
        print("Error: true labels differ between runs!")
        sys.exit(1)
    true = true1

    # Compute Youden thresholds
    fpr1, tpr1, thr1 = roc_curve(true, prob1)
    youden1 = tpr1 - fpr1
    thr_opt1 = thr1[np.argmax(youden1)]
    pred1 = (prob1 >= thr_opt1).astype(int)

    fpr2, tpr2, thr2 = roc_curve(true, prob2)
    youden2 = tpr2 - fpr2
    thr_opt2 = thr2[np.argmax(youden2)]
    pred2 = (prob2 >= thr_opt2).astype(int)

    print(f"Model1 Youden threshold: {thr_opt1:.4f}")
    print(f"Model2 Youden threshold: {thr_opt2:.4f}\n")

    acc1 = np.mean(pred1 == true) * 100
    print(f"Model 1 Youden accuracy: {acc1:.2f}%\n")

    acc2 = np.mean(pred2 == true) * 100
    print(f"Model 2 Youden accuracy: {acc2:.2f}%\n")

    # Prepare differences
    corr1 = (pred1 == true).astype(int)
    corr2 = (pred2 == true).astype(int)
    diffs = corr1 - corr2

    # Run TOST
    t1, p1, t2, p2, equivalent = tost_equivalence_test(diffs, delta, alpha)

    print("TOST Equivalence Test Results:")
    print(f"  t1 = {t1:.4f}, p1 = {p1:.4e}  # test mean > -delta")
    print(f"  t2 = {t2:.4f}, p2 = {p2:.4e}  # test mean < +delta")
    if equivalent:
        print(f"Equivalent within +/- {delta} at alpha = {alpha}")
    else:
        print(f"Not equivalent within +/- {delta} at alpha = {alpha}")

    # mean and standard error of diffs
    mean_d = np.mean(diffs)
    sd_d   = np.std(diffs, ddof=1)
    se     = sd_d / np.sqrt(len(diffs))

    # add plotting
    plot_tost_ci(mean_d, se, len(diffs), delta, alpha)

if __name__ == "__main__":
    main()
