import os
import sys
import cv2
import glob
import numpy as np
import caffe

def preprocess_image(img, input_shape):
    """
    Preprocess the input image:
      - Resize the image to the expected (width, height) from input_shape.
      - Normalize the pixel values to [0, 1].
      - Rearrange dimensions to (channels, height, width) as expected by Caffe.
    
    Parameters:
        img (numpy.ndarray): Input image in BGR format.
        input_shape (tuple): Expected input shape in the form (channels, height, width).
    
    Returns:
        numpy.ndarray: Preprocessed image.
    """
    # Extract expected height and width (ignore number of channels)
    _, expected_h, expected_w = input_shape
    # Resize the image (cv2.resize expects (width, height))
    img_resized = cv2.resize(img, (expected_w, expected_h))
    # Convert to float and normalize pixel values to [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0
    # Change image shape from (height, width, channels) to (channels, height, width)
    img_processed = img_norm.transpose(2, 0, 1)
    return img_processed

def main():
    if len(sys.argv) < 5:
        print("Usage: python filter_caffe.py <deploy.prototxt> <model.caffemodel> <png_folder> <activation_type>")
        print("activation_type: 'softmax' or 'sigmoid'")
        sys.exit(1)
    
    deploy_prototxt = sys.argv[1]
    caffe_model = sys.argv[2]
    png_folder = sys.argv[3]
    activation_type = sys.argv[4].lower()
    
    if activation_type not in ["softmax", "sigmoid"]:
        print("Error: activation_type must be either 'softmax' or 'sigmoid'.")
        sys.exit(1)
    
    # Set Caffe to GPU mode (adjust device settings as needed)
    caffe.set_mode_gpu()
    caffe.set_device(0)  # Change this if you wish to use a different GPU or multiple GPUs

    # Load the network in test mode
    net = caffe.Net(deploy_prototxt, caffe_model, caffe.TEST)
    
    # Determine the expected input shape and batch size
    # net.blobs['data'].data has shape (batch_size, channels, height, width)
    input_shape = net.blobs['data'].data.shape[1:]
    batch_size = net.blobs['data'].data.shape[0]
    
    # Gather list of all PNG files in the specified folder
    image_paths = glob.glob(os.path.join(png_folder, "*.png"))
    print(f"Found {len(image_paths)} PNG images in folder '{png_folder}'.")
    initial_count = len(image_paths)
    
    batch_data = []
    batch_files = []  # Keep track of corresponding file paths for each image in the batch
    
    for image_path in image_paths:
        # Do not process (or remove) images whose filename contains 'fovea'
        if "fovea" in os.path.basename(image_path).lower():
            print(f"Skipping image with 'fovea' in name: {image_path}")
            continue
        
        # Load the image using OpenCV (BGR format)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_path}. Skipping.")
            continue
        
        # Preprocess the image to match the network's expected input
        img_processed = preprocess_image(img, input_shape)
        
        batch_data.append(img_processed)
        batch_files.append(image_path)
        
        # When the batch is full, perform inference
        if len(batch_data) == batch_size:
            data_array = np.array(batch_data)
            net.blobs['data'].data[...] = data_array
            output = net.forward()
            
            # Process the model's output based on the activation type
            if activation_type == "sigmoid":
                # Assumes that the 'sigmoid' layer outputs a single probability per sample
                probs = output['sigmoid'].flatten()
                predictions = (probs > 0.5).astype(int)
            else:  # softmax
                # Assumes that the 'prob' layer outputs probabilities for each class
                probs = output['prob']
                predictions = np.argmax(probs, axis=1)
            
            # For each image in the batch, if predicted class is not 1, remove the file
            for file, pred in zip(batch_files, predictions):
                if pred != 1:
                    try:
                        os.remove(file)
                        print(f"Removed image: {file} (predicted class: {pred})")
                    except Exception as e:
                        print(f"Error removing file {file}: {e}")
            
            # Clear batch lists for the next set of images
            batch_data.clear()
            batch_files.clear()
    
    # Process any remaining images that did not fill an entire batch.
    if batch_data:
        current_batch_size = len(batch_data)
        # Caffe may require the full batch shape. Create an array of zeros for padding.
        padded_data = np.zeros((batch_size, *input_shape), dtype=np.float32)
        padded_data[:current_batch_size] = np.array(batch_data)
        net.blobs['data'].data[...] = padded_data
        output = net.forward()
        
        if activation_type == "sigmoid":
            probs = output['sigmoid'].flatten()
            predictions = (probs > 0.5).astype(int)
        else:
            probs = output['prob']
            predictions = np.argmax(probs, axis=1)
        
        for file, pred in zip(batch_files, predictions[:current_batch_size]):
            if pred != 1:
                try:
                    os.remove(file)
                    print(f"Removed image: {file} (predicted class: {pred})")
                except Exception as e:
                    print(f"Error removing file {file}: {e}")

    # Get final count
    final_count = len(glob.glob(os.path.join(png_folder, "*.png")))
    
    # Print start and end count on the same line
    print(f"Images before filtering: {initial_count}, after filtering: {final_count}")

if __name__ == '__main__':
    main()
