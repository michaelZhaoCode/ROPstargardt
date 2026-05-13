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
    """
    Runs the given Caffe model on test_lmdb and returns:
       - true_labels: ndarray of shape (N,)
       - probs:       ndarray of shape (N,), probability of the positive class
    """
    caffe.set_mode_gpu()
    caffe.set_devices([0,1])
    net = caffe.Net(deploy_prototxt, caffe_model, caffe.TEST)

    env = lmdb.open(test_lmdb, readonly=True)
    txn = env.begin()
    cursor = txn.cursor()

    batch_size = net.blobs['data'].data.shape[0]
    batch_data, batch_labels = [], []
    true_labels, probs = [], []

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
            else:  # softmax
                p = out['prob'][:, 1]

            true_labels.extend(batch_labels)
            probs.extend(p.tolist())

            batch_data.clear()
            batch_labels.clear()

    env.close()
    return np.array(true_labels), np.array(probs)

def paired_t_and_bootstrap(true, pred1, pred2, n_boot=5000, ci=95):
    corr1 = (pred1 == true).astype(int)
    corr2 = (pred2 == true).astype(int)
    diffs = corr1 - corr2
    mean_diff = diffs.mean()

    # Paired t‑test
    t_stat, p_value = stats.ttest_rel(corr1, corr2)

    # Bootstrap
    n = len(diffs)
    boot_means = np.empty(n_boot)
    rng = np.random.RandomState()
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_means[i] = diffs[idx].mean()
    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100 - (100-ci)/2)

    return t_stat, p_value, mean_diff, lower, upper, boot_means

def plot_bootstrap(boot_means, mean_diff, ci_lo, ci_hi, out_path="plots/bootstrap_diff.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.hist(boot_means, bins=50, edgecolor='black')
    plt.axvline(mean_diff, color='red', linestyle='--', label=f'Mean diff = {mean_diff:.4f}')
    plt.axvline(ci_lo, color='blue', linestyle=':', label=f'95% CI lower = {ci_lo:.4f}')
    plt.axvline(ci_hi, color='blue', linestyle=':', label=f'95% CI upper = {ci_hi:.4f}')
    plt.axvline(0, color='black', linestyle='-', label='Zero difference')
    plt.title('Bootstrap Distribution of Mean(per-sample correct1 - correct2)')
    plt.xlabel('Mean difference')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Bootstrap histogram saved to {out_path}")

def main():
    if len(sys.argv) != 6:
        print("Usage: python compare_models.py "
              "<deploy.prototxt> <model1.caffemodel> <model2.caffemodel> "
              "<test_lmdb> <activation_type>")
        sys.exit(1)

    deploy, model1, model2, lmdb_path, act = sys.argv[1:]
    print("Evaluating Model 1")
    true1, prob1 = evaluate_model(deploy, model1, lmdb_path, act)
    print("Evaluating Model 2")
    true2, prob2 = evaluate_model(deploy, model2, lmdb_path, act)

    assert np.array_equal(true1, true2), "True labels differ between runs!"
    true = true1

    # Compute optimal thresholds via Youden's index
    fpr1, tpr1, thr1 = roc_curve(true, prob1)
    youden1 = tpr1 - fpr1
    thr_opt1 = thr1[np.argmax(youden1)]
    pred1 = (prob1 >= thr_opt1).astype(int)
    print(f"Model 1 optimal threshold (Youden's): {thr_opt1:.4f}")

    fpr2, tpr2, thr2 = roc_curve(true, prob2)
    youden2 = tpr2 - fpr2
    thr_opt2 = thr2[np.argmax(youden2)]
    pred2 = (prob2 >= thr_opt2).astype(int)
    print(f"Model 2 optimal threshold (Youden's): {thr_opt2:.4f}")

    assert np.array_equal(true1, true2), "True labels differ between runs!"
    true = true1

    acc1 = np.mean(pred1 == true) * 100
    print(f"Model 1 Youden accuracy: {acc1:.2f}%\n")

    acc2 = np.mean(pred2 == true) * 100
    print(f"Model 2 Youden accuracy: {acc2:.2f}%\n")

    print("\nRunning paired t-test & bootstrap")
    t_stat, p_val, mean_diff, ci_lo, ci_hi, boot_means = paired_t_and_bootstrap(
        true, pred1, pred2, n_boot=5000, ci=95)

    print(f"\nPaired t-test:")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value     = {p_val:.4e}")

    print(f"\nMean(per-sample correct2 - correct1) = {mean_diff:.4f}")
    print(f"95% bootstrap CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

    if ci_lo > 0 or ci_hi < 0:
        print(" 0 is outside the 95% CI, significant difference.")
    else:
        print(" 0 is inside the 95% CI, no significant difference.")

    # Plot and save the bootstrap distribution
    plot_bootstrap(boot_means, mean_diff, ci_lo, ci_hi)

if __name__ == "__main__":
    main()
