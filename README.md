# Improving Regional OCT-Based Classification of Stargardt Disease Through GAN-Generated Synthetic Images

Stargardt disease is a progressive, inherited macular dystrophy characterized by retinal degeneration that typically originates in the fovea and expands outward. While deep learning models have shown promise in diagnosing retinal conditions through Optical Coherence Tomography (OCT), existing classifiers are often restricted to the central retina due to a scarcity of comprehensive, spatially diverse training data.

This study investigates whether generative AI—specifically Generative Adversarial Networks (GANs)—can bridge this gap by creating synthetic OCT imagery to enhance the diagnostic capabilities of deep learning models. By training a StyleGAN2-ADA model to generate realistic Stargardt OCT patches, the researchers evaluate whether supplementing or replacing traditional datasets can improve the accuracy and spatial generalizability of classifiers across the entire retina, potentially offering a robust solution to data limitations in clinical ophthalmology.

## File Breakdown
**Data Processing**
* **`data.py`**: Handles initial data preprocessing. It parses image metadata, extracts the optimal patches based on intensity and fovea location, and resizes them to 128x128.
* **`format_dataset.py`**: Partitions the extracted patches into a 5-fold cross-validation framework. It strictly enforces patient-level splitting to prevent spatial data leakage and automatically downsamples the normal (majority) class to maintain exact image-level class balance across all folds.
* **`filter_images.py`**: A utility script that uses a trained Caffe model to run inference on a folder of images and delete those that do not meet a specific predicted class.

**Caffe Network Configurations (`.prototxt`)**
* **`train_val_softmax.prototxt`**: Defines the VGG19 network architecture, data layers (LMDB format), and loss functions for the training and validation phases.
* **`deploy_softmax.prototxt`**: Defines the network architecture for deployment/inference (takes a single image as input rather than an LMDB batch).
* **`solver.prototxt`**: Contains the training hyperparameters (e.g., base learning rate of 0.0001, step decay, momentum, max iterations).

**Evaluation & Statistical Analysis**
* **`test_model.py`**: Evaluates a trained Caffe model against a test LMDB dataset. Outputs comprehensive metrics like Accuracy, Precision, Recall, Specificity, F1 Score, and ROC AUC based on Youden's optimal threshold.
* **`test_plot.py`**: Evaluates multiple models and generates line plots comparing their accuracies across different spatial locations on the retina.
* **`bootstrap_diff.py`**: Performs paired t-tests and bootstrap analyses (5000 iterations) to compare the performance differences between two models.
* **`equivalence_test.py`**: Runs a Two One-Sided Tests (TOST) procedure to determine statistical equivalence between model predictions. 

**Execution**
* **`run instructions.txt`**: Contains the exact Docker command to spin up the required NVIDIA Caffe environment and the command to initiate model training.

---

### **How to Replicate This Process**

To reproduce this study, a researcher should follow these steps:

1.  **Data Preparation (Real Data):** Run `data.py` on the raw OCT dataset to extract and format 128x128 patches.
2.  **Data Splitting & Balancing**: Run `format_dataset.py` to organize the extracted patches into a balanced 5-fold cross-validation structure. This ensures patient-level isolation between training, validation, and testing sets, and establishes an equal ratio of Normal to Stargardt patches.
3.  **GAN Augmentation (External to Codebase):**
    * *Note on running the GAN:* To generate the synthetic data, you will need to train a StyleGAN2-ADA model on the real Stargardt OCT patches extracted in Step 1. Once the model is trained, use it to generate synthetic patches. Merge these synthetic images with your real training dataset (to augment or replace data as described in the paper). 
    * Convert the final merged dataset of images into LMDB format (the standard data format expected by `train_val_softmax.prototxt`).
4.  **Environment Setup:**
    * Ensure Docker with GPU support is installed.
    * Run the container command from `run instructions.txt` to launch the `nvcr.io/nvidia/caffe:20.03-py3` environment.
    * Install required Python packages inside the container: `pip install lmdb scikit-learn torchvision pandas matplotlib`.
5.  **Model Training:**
    * Execute the training command: `caffe train --solver=solver.prototxt --gpu=all --weights=VGG19.caffemodel` (ensure the base VGG19 weights are downloaded first).
6.  **Testing & Visualization:**
    * Run `test_model.py` using your `.caffemodel` snapshot and `deploy_softmax.prototxt` to get standard classification metrics.
    * Run `test_plot.py` to generate visual comparisons of accuracy across different retinal locations.
7.  **Statistical Validation:**
    * To prove the efficacy of the GAN augmentation, run `bootstrap_diff.py` and `equivalence_test.py` using the prediction outputs from the baseline model versus the GAN-augmented model.
