import concurrent.futures
import numpy as np
import cv2
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from Convolution import flatten, Conv2D, pool2d
from RGBtoGray import rgb2gray
from preprocessing import resize_image, image_to_array, change_color_to_rgb
from training import make_predictions
from io import BytesIO
import threading
import time  # Import time module

def process_and_save_single_image(image_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)  # Reduced iterations
    sure_bg = cv2.dilate(bin_img, kernel, iterations=1)  # Reduced iterations
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)  # Reduced mask size
    ret, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.resize(markers, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    markers = cv2.watershed(img, markers)
    labels = np.unique(markers)
    coins = []
    for label in labels[2:]:
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coins.append(contours[0])
    img = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)
    img = change_color_to_rgb(img, markers, labels)
    
    # Resize image to 224x224 after background removal
    img_resized = cv2.resize(img, (224, 224))
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    processed_output_path = os.path.join(output_path, f"{base_filename}_removal.jpg")
    cv2.imwrite(processed_output_path, img_resized)
    print(f"Processed and saved image to {processed_output_path}")
    return processed_output_path

def draw_bounding_box(image_path, bounding_box, label, output_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Resize image to 224x224 for consistency
    image = cv2.resize(image, (224, 224))
    
    # Unpack bounding box coordinates
    x, y, w, h = bounding_box
    
    # Draw bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Thinner line for small image
    
    # Put label with appropriate size for 224x224 image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Smaller font
    thickness = 1     # Thinner text
    color = (0, 255, 0)
    
    # Position label above bounding box
    label_y = max(y - 5, 15)  # Ensure label is visible
    cv2.putText(image, label, (x, label_y), font, font_scale, color, thickness)
    
    # Save the image with bounding box
    cv2.imwrite(output_path, image)
    print(f"Saved detection result to {output_path}")

def get_similar_pixels(image_path, prediction):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure prediction is a single value
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.item()
    
    # Convert prediction to the same type as image
    prediction = np.array(prediction, dtype=image.dtype)
    
    # Find pixels similar to the prediction
    similar_pixels = np.where(image == prediction)
    
    # Check if similar_pixels contains valid data
    if similar_pixels[0].size == 0 or similar_pixels[1].size == 0:
        return np.array([])

    # Store similar pixel coordinates in an array
    similar_pixels_array = np.array(list(zip(similar_pixels[1], similar_pixels[0])))
    
    return similar_pixels_array

def print_similar_pixels(similar_pixels_array):
    # Print similar pixel coordinates
    for x, y in similar_pixels_array:
        print(f"Similar pixel at ({x}, {y})")

def get_bounding_box_from_pixels(similar_pixels_array):
    if similar_pixels_array.size == 0:
        raise ValueError("No similar pixels found.")
    
    # Get the minimum and maximum coordinates
    min_x, min_y = np.min(similar_pixels_array, axis=0)
    max_x, max_y = np.max(similar_pixels_array, axis=0)
    
    # Calculate width and height
    width = max_x - min_x
    height = max_y - min_y
    
    return (min_x, min_y, width, height)   
 # Get similar pixels
    similar_pixels_array = get_similar_pixels(image_path, prediction)

    # Check if similar pixels are found
    if similar_pixels_array.size == 0:
        print("No similar pixels found.")
    else:
        # Generate bounding box coordinates from similar pixels
        bounding_box = get_bounding_box_from_pixels(similar_pixels_array)

        # Draw bounding box on the original image with label
        draw_bounding_box(image_path, bounding_box, f'Label: {prediction}')

def process_all_images_in_directory(directory_path):
    image_tensors = []
    image_paths = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            print(f"Loading image: {image_path}")
            image = Image.open(image_path).convert('L')
            image = np.array(image)
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensors.append(image_tensor)
            image_paths.append(image_path)
    
    if image_tensors:
        # Combine all image tensors into a single batch tensor
        batch_tensor = torch.cat(image_tensors, dim=0)
        print(f"Processing batch of {len(image_tensors)} images")
        test_conv2d_with_batch(batch_tensor, image_paths)

def test_conv2d_with_batch(batch_tensor, image_paths):
    padding = 1
    pool_kernel_size = (2, 2)  # Ensure this is a tuple

    # Define emboss kernel
    emboss_kernel = torch.tensor([[[[-2, -1, 0],
                                    [-1,  1, 1],
                                    [ 0,  1, 2]]]], dtype=torch.float32)

    # Create Conv2D instances for three layers
    conv2d_1 = Conv2D(emboss_kernel, padding)
    conv2d_2 = Conv2D(emboss_kernel, padding)
    conv2d_3 = Conv2D(emboss_kernel, padding)

    # Measure processing time
    start_time = time.time()

    # Apply Conv2D and pooling layers
    print("proses conv1d")
    Y = conv2d_1(batch_tensor)
    print("proses pooling")
    Y = pool2d(Y, pool_kernel_size)
    print("proses conv2d")
    Y = conv2d_2(Y)
    print("proses pooling")
    Y = pool2d(Y, pool_kernel_size)
    print("proses conv3d")
    Y = conv2d_3(Y)
    print("proses pooling")
    Y = pool2d(Y, pool_kernel_size)

    # Flatten the output
    Y_flattened = flatten(Y)

    # Transpose the flattened output for predictions
    Y_transposed = Y_flattened.transpose(0, 1)

    # Make predictions (simple classification based on feature mean)
    # For a proper implementation, you would load trained weights W1, b1, W2, b2
    # and use: predictions = make_predictions(Y_transposed, W1, b1, W2, b2)
    
    # Simple prediction based on feature statistics (placeholder)
    feature_means = torch.mean(Y_transposed, dim=0)
    predictions = (feature_means > torch.median(feature_means)).int().numpy()

    # Print processing time
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")

    # Display results and save output files
    for i, (image_path, prediction) in enumerate(zip(image_paths, predictions)):
        print(f"Prediction for {image_path}: {prediction}")
        
        # Save prediction result to text file
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        prediction_file = os.path.join("c:\\ujicobaprogram\\Deteksi Vannamei\\detection_results", f"{base_filename}_prediction.txt")
        
        with open(prediction_file, 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Class: {'Udang_Vannamei' if prediction == 0 else 'Bukan_Udang'}\n")
            f.write(f"Processing time: {end_time - start_time:.2f} seconds\n")
        
        # Copy original image to results folder with prediction label
        output_image_path = os.path.join("c:\\ujicobaprogram\\Deteksi Vannamei\\detection_results", 
                                       f"{base_filename}_{'Vannamei' if prediction == 0 else 'BukanUdang'}.jpg")
        
        # Load and save image with prediction label
        original_image = cv2.imread(image_path)
        if original_image is not None:
            # Resize image to 224x224 to match training data size
            resized_image = cv2.resize(original_image, (224, 224))
            
            # Add prediction text to image with appropriate size for 224x224
            label_text = f"{'Vannamei' if prediction == 0 else 'Bukan Udang'}"
            
            # Font parameters optimized for 224x224 image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # Smaller font for 224x224 image
            thickness = 1     # Thinner text
            color = (0, 255, 0)  # Green color
            
            # Get text size to position it properly
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Position text at top-left with small margin
            x = 5
            y = text_height + 5
            
            # Add background rectangle for better text visibility
            cv2.rectangle(resized_image, (x-2, y-text_height-2), (x+text_width+2, y+baseline+2), (0, 0, 0), -1)
            
            # Add text on top of background
            cv2.putText(resized_image, label_text, (x, y), font, font_scale, color, thickness)
            
            cv2.imwrite(output_image_path, resized_image)
            print(f"Saved result image to: {output_image_path}")
            print(f"Saved prediction text to: {prediction_file}")
        
        print("-" * 50)
    
    # Save detection summary
    summary_file = os.path.join("c:\\ujicobaprogram\\Deteksi Vannamei\\detection_results", "detection_summary.txt")
    vannamei_count = sum(1 for p in predictions if p == 0)
    bukan_udang_count = sum(1 for p in predictions if p == 1)
    
    with open(summary_file, 'w') as f:
        f.write("DETECTION SUMMARY\n")
        f.write("================\n\n")
        f.write(f"Total images processed: {len(predictions)}\n")
        f.write(f"Udang Vannamei detected: {vannamei_count}\n")
        f.write(f"Bukan Udang detected: {bukan_udang_count}\n")
        f.write(f"Processing time: {end_time - start_time:.2f} seconds\n\n")
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for image_path, prediction in zip(image_paths, predictions):
            filename = os.path.basename(image_path)
            result = 'Udang_Vannamei' if prediction == 0 else 'Bukan_Udang'
            f.write(f"{filename}: {result}\n")
    
    print(f"Detection summary saved to: {summary_file}")
    print(f"Total: {len(predictions)} images | Vannamei: {vannamei_count} | Bukan Udang: {bukan_udang_count}")

def main():
    directory_path = "c:\\ujicobaprogram\\Deteksi Vannamei\\data"
    output_path = "c:\\ujicobaprogram\\Deteksi Vannamei\\detection_results"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print("CNN Vannamei Detection Started...")
    print(f"Input directory: {directory_path}")
    print(f"Output directory: {output_path}")
    print("-" * 50)
        
    process_all_images_in_directory(directory_path)

if __name__ == "__main__":
    main()