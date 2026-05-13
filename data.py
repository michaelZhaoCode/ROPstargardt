import os
import numpy as np
import random
import shutil
from tqdm import tqdm
from PIL import Image


def parse_image_metadata(filename):
    """
    Parse metadata from the image filename.

    Args:
        filename (str): The filename of the image.

    Returns:
        dict: A dictionary containing metadata fields.
    """
    parts = filename.split('-')
    stack_id = parts[2]
    num_in_stack = int(parts[3])
    angle = int(parts[4].split('_')[0])
    return {
        "stack_id": stack_id,
        "num_in_stack": num_in_stack,
        "angle": angle
    }


def process_stargardt_images(base_dir, output_dir, processed_image_dir, radials_per_patient=50, non_radials_per_patient=50):
    """
    Process images for each patient by sampling and calling preprocessing and patching functions,
    and move the base images to a specified folder.

    Args:
        base_dir (str): Path to the base directory containing patient subdirectories.
        output_dir (str): Path to save the processed patches.
        processed_image_dir (str): Path to move the processed base images.
        radials_per_patient (int): Number of radial images to sample per patient.
        non_radials_per_patient (int): Number of non-radial images to sample per patient.
    """
    # Ensure the processed image directory exists
    os.makedirs(processed_image_dir, exist_ok=True)

    for patient_dir in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        images_path = os.path.join(patient_path, "images")
        if not os.path.exists(images_path):
            print(f"Skipping {patient_dir}: No 'images' directory found.")
            continue

        radial_images = []
        non_radial_images = []
        valid_stacks = set()

        # Collect radial and non-radial images, skipping invalid stacks
        for image_file in os.listdir(images_path):
            if not image_file.endswith(".png"):
                continue

            metadata = parse_image_metadata(image_file)
            stack_id = metadata["stack_id"]

            # Skip all images in stacks with num_in_stack > 61
            if metadata["num_in_stack"] > 61:
                print(f"Found invalid stack, {stack_id}")
                valid_stacks.discard(stack_id)
                continue

            valid_stacks.add(stack_id)

            # Categorize images as radial or non-radial
            if metadata["angle"] == 0:
                non_radial_images.append(os.path.join(images_path, image_file))
            else:
                radial_images.append(os.path.join(images_path, image_file))

        # Filter images by valid stacks
        radial_images = [img for img in radial_images if parse_image_metadata(os.path.basename(img))["stack_id"] in valid_stacks]
        non_radial_images = [img for img in non_radial_images if parse_image_metadata(os.path.basename(img))["stack_id"] in valid_stacks]

        # Calculate adjusted sample sizes
        available_radials = len(radial_images)
        available_non_radials = len(non_radial_images)

        adjusted_radials = radials_per_patient
        adjusted_non_radials = non_radials_per_patient

        # If one pool cannot fulfill the requested size, add the shortfall to the other pool
        if available_radials < radials_per_patient:
            adjusted_non_radials += radials_per_patient - available_radials
        elif available_non_radials < non_radials_per_patient:
            adjusted_radials += non_radials_per_patient - available_non_radials

        adjusted_radials = min(available_radials, adjusted_radials)
        adjusted_non_radials = min(available_non_radials, adjusted_non_radials)

        # Sample from pools with adjusted sizes
        sampled_radials = random.sample(radial_images, adjusted_radials)
        sampled_non_radials = random.sample(non_radial_images, adjusted_non_radials)

        # Process sampled images
        for idx, image_path in enumerate(tqdm(sampled_radials + sampled_non_radials, desc="Processing images")):
            is_radial = parse_image_metadata(os.path.basename(image_path))["angle"] != 0

            # Preprocess the image
            preprocessed_image = preprocess_image_for_oct(image_path, stargardt=True)

            # Process the patches
            process_oct(
                preprocessed_image,
                output_dir=output_dir,
                num_croppings=15,
                patch_height_percent=0.3,
                stargardt=True,
                fovea=is_radial,
                patient_id=patient_dir + f"_{idx}"
            )

            # Move the base image to the processed_image_dir
            relative_path = os.path.relpath(image_path, base_dir)
            dest_path = os.path.join(processed_image_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(image_path, dest_path)

    print("Processing completed successfully.")


def process_normal_images(base_dir, output_dir, processed_image_dir, limit: int = 10000):
    """
    Process all normal images by calling preprocessing and patching functions,
    and move the base images to a specified folder.

    Args:
        base_dir (str): Path to the directory containing all normal images.
        output_dir (str): Path to save the processed patches.
        processed_image_dir (str): Path to move the processed base images.
        limit:
    """
    # Ensure the processed image directory exists
    os.makedirs(processed_image_dir, exist_ok=True)

    # Process all normal images in the base directory
    for image_file in tqdm(os.listdir(base_dir)[:limit], desc="Processing images"):
        if not image_file.endswith(".jpeg"):
            continue

        # Parse patient_id from the filename
        if not image_file.startswith("NORMAL-"):
            print(f"Skipping invalid file: {image_file}")
            continue

        patient_id = image_file.split('-')[1]
        image_num = image_file.split('-')[2].split(".")[0]

        # Full path to the image file
        image_path = os.path.join(base_dir, image_file)

        # Preprocess the image
        preprocessed_image = preprocess_image_for_oct(image_path, stargardt=False)

        # Process the patches
        process_oct(
            preprocessed_image,
            output_dir=output_dir,
            stargardt=False,
            fovea=True,
            patient_id=patient_id + f"_{image_num}"
        )

        # Move the base image to the processed_image_dir
        dest_path = os.path.join(processed_image_dir, image_file)
        shutil.move(image_path, dest_path)

    print("Processing of normal images completed successfully.")


def preprocess_image_for_oct(image_path, stargardt=True):
    """
    Preprocess a PNG image by cropping, resizing, and converting it to a format suitable for the process_oct function.

    Args:
        image_path (str): Path to the input PNG image.
        stargardt (bool): If the image is Stargardt.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    # Load the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    if stargardt:
        # Crop the top 220 pixels and the bottom 340 pixels
        cropped_image = image.crop((0, 220, image.width, image.height - 340))

        # Convert the image to a NumPy array
        image_array = np.array(cropped_image)
    else:
        image_array = np.array(image)

    return image_array


def process_oct(image, num_croppings=10, patch_height_percent=0.25, output_dir="data", stargardt=True, fovea=True, patient_id="unknown"):
    """
    Process an OCT image to extract square patches based on maximum grayscale sum along specified height percentages,
    resize to 128x128, and save them in the appropriate directory with additional metadata in the file name.

    Args:
        image (numpy.ndarray): Input grayscale image.
        num_croppings (int): Number of patches to extract along the width.
        patch_height_percent (float): Height of each patch as a percentage of the image height.
        output_dir (str): Base directory to save the resulting cropped patches.
        stargardt (bool): If True, save in `data/stargardt`; otherwise, `data/normal`.
        fovea (bool): If True, include `_fovea` in file names for patches containing the fovea.
        patient_id (str): Patient ID to include in file names.

    Returns:
        bool: Success status.
    """
    # Determine patch dimensions based on the percentage of image height
    image_height, image_width = image.shape
    patch_height = int(image_height * patch_height_percent)
    patch_width = patch_height  # Ensure patches are square

    # Divide the image into fixed cropping sections along its width
    x_positions = np.linspace(0, image_width - patch_width, num=num_croppings, endpoint=True, dtype=int)

    # Set up the output directory
    category = "stargardt" if stargardt else "normal"
    output_dir = os.path.join(output_dir, category)
    os.makedirs(output_dir, exist_ok=True)

    # Fovea detection (find the fovea column in the image)
    middle_third_start = image_width // 3
    middle_third_end = 2 * image_width // 3
    fovea_x = None
    fovea_curve = None
    for x in range(middle_third_start, middle_third_end):
        column_profile = np.sum(image[:, x])
        if fovea_curve is None or column_profile < fovea_curve:
            fovea_curve = column_profile
            fovea_x = x

    # Validate that the fovea is in the middle third of the image
    if fovea_x is None:
        fovea = False

    # Define height constraints to avoid borders
    height_margin = patch_height // 4
    valid_y_start = height_margin
    valid_y_end = image_height - patch_height - height_margin

    for i, x in enumerate(x_positions):
        # Slide along the valid height range to find the optimal patch position
        max_sum = 0
        optimal_y = 0

        for y in range(valid_y_start, valid_y_end + 1):
            patch_sum = np.sum(image[y:y + patch_height, x:x + patch_width])
            if patch_sum > max_sum:
                max_sum = patch_sum
                optimal_y = y

        # Extract the patch at the optimal position
        patch = image[optimal_y:optimal_y + patch_height, x:x + patch_width]

        # Resize the patch to 128x128
        patch_image = Image.fromarray(patch)
        patch_image_resized = patch_image.resize((128, 128), Image.LANCZOS)

        # Assert that the resized patch is not larger than the original
        assert patch_image_resized.size[0] <= patch_image.size[0] and patch_image_resized.size[1] <= patch_image.size[1]

        # Determine if the patch contains the fovea
        threshold = 0.05
        if x + patch_width * threshold <= fovea_x < x + patch_width * (1 - threshold) and fovea:
            fovea_flag = "_fovea"
        else:
            fovea_flag = ""

        # Save the patch
        patch_filename = f"{patient_id}_{i + 1}{fovea_flag}.png"
        patch_path = os.path.join(output_dir, patch_filename)
        patch_image_resized.save(patch_path)

    return True


if __name__ == "__main__":
    process_normal_images(base_dir="data/unprocessed_normals/NORMAL", output_dir="data", processed_image_dir="data/processed_normals")
