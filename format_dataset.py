import os
import shutil
import random
from sklearn.model_selection import KFold

# Define dataset paths
DATASET_DIR = "fovea"  # Root dataset directory
CLASS_0_DIR = os.path.join(DATASET_DIR, "normal")  # Class 0: normal
CLASS_1_DIR = os.path.join(DATASET_DIR, "stargardt")  # Class 1: stargardt
OUTPUT_DIR = "formatting"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get sorted image lists for each class
class_0_images = sorted(os.listdir(CLASS_0_DIR))
class_1_images = sorted(os.listdir(CLASS_1_DIR))

# --------------------------
# Helper function to extract patient ID from the image filename.
def get_patient_id(filename):
    # Assumes filename format: patientid_num_num_text.png
    return filename.split('_')[0]


# --------------------------
# Group images by patient id for each class

def group_by_patient(image_list):
    groups = {}
    for img in image_list:
        pid = get_patient_id(img)
        groups.setdefault(pid, []).append(img)
    return groups


class_1_groups = group_by_patient(class_1_images)
class_0_groups = group_by_patient(class_0_images)

# Get the unique patient ids
class_1_pids = sorted(list(class_1_groups.keys()))
class_0_pids = sorted(list(class_0_groups.keys()))

# --------------------------
# Split class 1 (stargardt) patients into 5 folds.
kf_class1 = KFold(n_splits=5, shuffle=True, random_state=42)
folds_class1 = []  # Each element will be a list of images for that fold (from class 1)
for _, val_idx in kf_class1.split(class_1_pids):
    fold_imgs = []
    # Get patient ids for this fold
    fold_pids = [class_1_pids[i] for i in val_idx]
    # Collect all images for these patients
    for pid in fold_pids:
        fold_imgs.extend(class_1_groups[pid])
    folds_class1.append(fold_imgs)

# --------------------------
# Split class 0 (normal) patients into 5 folds.
kf_class0 = KFold(n_splits=5, shuffle=True, random_state=42)
folds_class0 = []  # Each element will be a list of images for that fold (from class 0)
for _, val_idx in kf_class0.split(class_0_pids):
    fold_imgs = []
    fold_pids = [class_0_pids[i] for i in val_idx]
    for pid in fold_pids:
        fold_imgs.extend(class_0_groups[pid])
    folds_class0.append(fold_imgs)

# --------------------------
# Balance the number of images in each fold between the two classes.
# We assume that class 1 is the limiting factor (fewer images per fold) and therefore,
# for each fold, if class 0 has more images, we randomly sample down to match.
balanced_folds = []  # List of tuples: (class0_images, class1_images) for each fold
for i in range(5):
    imgs1 = folds_class1[i]
    imgs0 = folds_class0[i]
    n1 = len(imgs1)
    n0 = len(imgs0)

    # Downsample the larger set so that each fold is balanced.
    if n0 > n1:
        imgs0 = random.sample(imgs0, n1)
        print(f"Downsampled norms by {abs(n1 - n0)}")
    elif n1 > n0:
        imgs1 = random.sample(imgs1, n0)
        print(f"Downsampled stargardt by {abs(n1 - n0)}")
    print(f"Fold {i + 1}: Class 0 Count = {len(imgs0)}, Class 1 Count = {len(imgs1)}\n")
    balanced_folds.append((imgs0, imgs1))

# At this point, balanced_folds is a list of 5 folds. Each fold is a tuple:
#   (list_of_class0_images, list_of_class1_images)
# and for each fold len(list_of_class0_images) == len(list_of_class1_images).

# --------------------------
# Create test set and cross-validation folds.
# Here, we designate the first fold as the test set and the remaining four as CV folds.

# Create test folder structure
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images")
TEST_LABELS_FILE = os.path.join(TEST_DIR, "test.txt")
os.makedirs(TEST_IMG_DIR, exist_ok=True)

# Use fold 0 as the test set
test_class0, test_class1 = balanced_folds[0]
test_set = [(img, 0) for img in test_class0] + [(img, 1) for img in test_class1]

# Copy test images and create label file
with open(TEST_LABELS_FILE, "w") as f:
    for img, label in test_set:
        src_dir = CLASS_0_DIR if label == 0 else CLASS_1_DIR
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(TEST_IMG_DIR, img)
        shutil.copy(src_path, dst_path)
        f.write(f"images/{img} {label}\n")
print(f"Test set saved in {TEST_DIR}")

# --------------------------
# For the remaining folds, create cross-validation splits.
# We use folds 1-4, and for each CV fold we use one fold as validation and the others as training.
for cv_fold in range(1, 5):
    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{cv_fold}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_txt = os.path.join(fold_dir, "train.txt")
    val_txt = os.path.join(fold_dir, "val.txt")

    # Prepare validation set for this fold (from balanced_folds[cv_fold])
    val_class0, val_class1 = balanced_folds[cv_fold]
    val_set = [(img, 0) for img in val_class0] + [(img, 1) for img in val_class1]

    # For training, combine all other folds (using folds 1-4 except the current fold)
    train_class0 = []
    train_class1 = []
    for i in range(1, 5):
        if i == cv_fold:
            continue
        imgs0, imgs1 = balanced_folds[i]
        train_class0.extend(imgs0)
        train_class1.extend(imgs1)
    train_set = [(img, 0) for img in train_class0] + [(img, 1) for img in train_class1]

    # Copy training images and write training label file
    with open(train_txt, "w") as f_train:
        for img, label in train_set:
            src_dir = CLASS_0_DIR if label == 0 else CLASS_1_DIR
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(train_dir, img)
            shutil.copy(src_path, dst_path)
            f_train.write(f"train/{img} {label}\n")

    # Copy validation images and write validation label file
    with open(val_txt, "w") as f_val:
        for img, label in val_set:
            src_dir = CLASS_0_DIR if label == 0 else CLASS_1_DIR
            src_path = os.path.join(src_dir, img)
            dst_path = os.path.join(val_dir, img)
            shutil.copy(src_path, dst_path)
            f_val.write(f"val/{img} {label}\n")

    print(f"Fold {cv_fold} saved in {fold_dir}")

print("Balanced cross-validation dataset created successfully!")
