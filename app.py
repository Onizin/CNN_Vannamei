from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
import torch
import cv2
from PIL import Image, UnidentifiedImageError  # Import UnidentifiedImageError
from Convolution import Conv2D, pool2d, flatten
from preprocessing import resize_image, image_to_array, change_color_to_rgb
from training import make_predictions
from io import BytesIO
import threading
import time  # Import time module
import concurrent.futures  # Import concurrent.futures for parallel processing

app = Flask(__name__)
# # dataset_path = "/media/ai/Micro/Coding/vannamei/ujicoba MLP/gambarinput/preprocessing/resize_grayscale"
# dataset_path = "E:\\vannamei\\ujicoba MLP\\gambarinput\\preprocessing\\resize_grayscale"

def denoise_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    denoised_image = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoised_image_path = os.path.join(output_path, f"denoised_{os.path.basename(image_path)}")
    cv2.imwrite(denoised_image_path, denoised_image)
    print(f"Denoised and saved image to {denoised_image_path}")
    return denoised_image_path

def process_and_save_single_image(image_path, output_path, output_noise, grayscale_output_path):
    # Apply denoising before background removal
    denoised_image_path = denoise_image(image_path, output_noise)
    if denoised_image_path is None:
        return None
    
    img = cv2.imread(denoised_image_path, cv2.IMREAD_COLOR)
    
    # Background removal process
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)  # Reduced iterations
    sure_bg = cv2.dilate(bin_img, kernel, iterations=2)  # Reduced iterations
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
    
    # Convert to grayscale and resize after background removal
    gray_resized_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_resized_path = os.path.join(grayscale_output_path, f"grayscale_{os.path.basename(image_path)}")
    cv2.imwrite(gray_resized_path, gray_resized_img)
    print(f"Grayscale and resized image saved to {gray_resized_path}")
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    processed_output_path = os.path.join(output_path, f"{base_filename}_removal.jpg")
    cv2.imwrite(processed_output_path, img_resized)
    print(f"Processed and saved image to {processed_output_path}")
    return processed_output_path

def process_images_in_parallel(image_paths, output_path, output_noise, grayscale_output_path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_and_save_single_image, image_path, output_path, output_noise, grayscale_output_path) for image_path in image_paths]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results

def draw_bounding_box(image_path, bounding_box, label, output_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Unpack bounding box coordinates
    x, y, w, h = bounding_box
    
    # Draw bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Put label inside the bounding box
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
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
    if (similar_pixels[0].size == 0 or similar_pixels[1].size == 0):
        return np.array([])

    # Store similar pixel coordinates in an array
    similar_pixels_array = np.array(list(zip(similar_pixels[1], similar_pixels[0])))
    
    return similar_pixels_array

def print_similar_pixels(similar_pixels_array):
    # Print similar pixel coordinates
    for x, y in similar_pixels_array:
        print(f"Similar pixel at ({x, y})")

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
    target_size = (224, 224)  # Define the target size for resizing
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            print(f"Loading image: {image_path}")
            try:
                image = Image.open(image_path).convert('L')
            except UnidentifiedImageError:
                print(f"UnidentifiedImageError: Cannot open image {image_path}")
                continue
            image = image.resize(target_size)  # Resize image to target size
            image = np.array(image)
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            image_tensors.append(image_tensor)
            image_paths.append(image_path)
    
    if image_tensors:
        # Combine all image tensors into a single batch tensor
        batch_tensor = torch.cat(image_tensors, dim=0)
        print(f"Processing batch of {len(image_tensors)} images")
        image_paths, predictions = test_conv2d_with_batch(batch_tensor, image_paths)
        return image_paths, predictions
    
    return image_paths, []  # Ensure the function returns the list of image paths and predictions

def save_tensor(tensor, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    torch.save({'tensor': tensor, 'filename': filename}, file_path)
    print(f"Saved tensor and filename to {file_path}")

def test_conv2d_with_batch(batch_tensor, image_paths):
    global stop_processing
    padding = 1
    pool_kernel_size = (2, 2)  # Ensure this is a tuple
    pool_stride = (2, 2)  # Ensure this is a tuple

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

    if stop_processing:
        return

    # Apply Conv2D and pooling layers
    print("proses conv1d")
    Y = conv2d_1(batch_tensor)
    save_tensor(Y, 'conv2d_1.pt', feature_extract_folder)
    if stop_processing:
        return

    print("proses pooling")
    Y = pool2d(Y, pool_kernel_size, 'max', pool_stride)
    save_tensor(Y, 'pool_1.pt', feature_extract_folder)
    if stop_processing:
        return

    print("proses conv2d")
    Y = conv2d_2(Y)
    save_tensor(Y, 'conv2d_2.pt', feature_extract_folder)
    if stop_processing:
        return

    print("proses pooling")
    Y = pool2d(Y, pool_kernel_size, 'max', pool_stride)
    save_tensor(Y, 'pool_2.pt', feature_extract_folder)
    if stop_processing:
        return

    print("proses conv3d")
    Y = conv2d_3(Y)
    save_tensor(Y, 'conv2d_3.pt', feature_extract_folder)
    if stop_processing:
        return

    print("proses pooling")
    Y = pool2d(Y, pool_kernel_size, 'max', pool_stride)
    save_tensor(Y, 'pool_3.pt', feature_extract_folder)
    if stop_processing:
        return

    # Flatten the output
    Y_flattened = flatten(Y)

    # Transpose the flattened output for predictions
    Y_transposed = Y_flattened.transpose(0, 1)
    file_trainsave = 'T:\\vannamei\\ujicoba MLP\\feature_Extract\\z_train\\train.pt'
    # file_trainsave = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/z_train/train.pt'
    checkpoint = torch.load(file_trainsave)
    W1, b1, W2, b2 = checkpoint['W1'], checkpoint['b1'], checkpoint['W2'], checkpoint['b2']

    # Make predictions
    predictions = make_predictions(Y_transposed, W1, b1, W2, b2)

    # Print processing time
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")

    # Display results
    for image_path, prediction in zip(image_paths, predictions):
        confidence = torch.softmax(torch.tensor(prediction, dtype=torch.float32), dim=0).max().item() * 100
        result = "Bukan Udang Vannamei" if prediction == 1 else "Udang Vannamei"
        print(f"Prediction for {image_path}: {result} with {confidence:.2f}% confidence")

    return image_paths, predictions


# Define the path to save processed images
processed_output_folder = 'T:\\vannamei\\ujicoba MLP\\training_uji\\prediksi'
uploaded_files_folder = 'T:\\vannamei\\ujicoba MLP\\training_uji\\uploaded_files'
detection_output_folder = 'T:\\vannamei\\ujicoba MLP\\training_uji\\deteksi'
denoized_file_folder = 'T:\\vannamei\\ujicoba MLP\\training_uji\\denoized'
grayscale_output_folder = 'T:\\vannamei\\ujicoba MLP\\training_uji\\grayscale'
feature_extract_folder = 'T:\\vannamei\\ujicoba MLP\\training_uji\\feature_extract'

# processed_output_folder = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/training_uji/prediksi'
# uploaded_files_folder = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/training_uji/uploaded_files'
# detection_output_folder = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/training_uji/deteksi'
# denoized_file_folder = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/training_uji/denoized'
# grayscale_output_folder = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/training_uji/grayscale'
# feature_extract_folder = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/training_uji/feature_extract'
# Global flag to stop processing
stop_processing = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_files', methods=['POST'])
def upload_files():
    global stop_processing
    stop_processing = False  # Reset stop flag
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'})
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No selected files'})
    
    folder_path = uploaded_files_folder
    os.makedirs(folder_path, exist_ok=True)
    
    # Clear the folder before uploading new files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Clear the processed output folder before saving new processed images
    for filename in os.listdir(processed_output_folder):
        file_path = os.path.join(processed_output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Clear the denoized folder before saving new denoized images
    for filename in os.listdir(denoized_file_folder):
        file_path = os.path.join(denoized_file_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Clear the grayscale folder before saving new grayscale images
    for filename in os.listdir(grayscale_output_folder):
        file_path = os.path.join(grayscale_output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    uploaded_files = []
    image_paths = []
    for file in files:
        file_path = os.path.join(folder_path, file.filename)
        file.save(file_path)
        image_paths.append(file_path)
    
    # Process images in parallel
    processed_image_paths = process_images_in_parallel(image_paths, processed_output_folder, denoized_file_folder, grayscale_output_folder)
    
    for file, processed_image_path in zip(files, processed_image_paths):
        uploaded_files.append({
            'raw_image_url': f'/uploads/{file.filename}',
            'processed_image_url': f'/uploads/{os.path.basename(processed_image_path)}',
            'filename': file.filename
        })

    return jsonify(uploaded_files)

@app.route('/process_files', methods=['POST'])
def process_files():
    global stop_processing
    stop_processing = False  # Reset stop flag

    # Clear the detection output folder before saving new detection results
    for filename in os.listdir(detection_output_folder):
        file_path = os.path.join(detection_output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Process all images in the prediksi directory
    image_paths, predictions = process_all_images_in_directory(processed_output_folder)
    
    # Collect processed images and their predictions
    processed_images = []
    processed_filenames = set()  # Set to track processed filenames
    for image_path, prediction in zip(image_paths, predictions):
        if stop_processing:
            break  # Stop processing if the stop flag is set
        filename = os.path.basename(image_path)
        if filename in processed_filenames:
            continue  # Skip if the file has already been processed
        processed_filenames.add(filename)
        result = "Bukan Udang Vannamei" if prediction == 1 else "Udang Vannamei"
        
        # Get similar pixels
        similar_pixels_array = get_similar_pixels(image_path, prediction)
        if similar_pixels_array.size == 0:
            print("No similar pixels found.")
            bounding_box = (0, 0, 0, 0)  # Default bounding box if no similar pixels found
        else:
            # Generate bounding box coordinates from similar pixels
            bounding_box = get_bounding_box_from_pixels(similar_pixels_array)
        
        # Draw bounding box on the original image with label
        detection_output_path = os.path.join(detection_output_folder, filename)
        draw_bounding_box(image_path, bounding_box, result, detection_output_path)
        
        processed_images.append({
            'raw_image_url': f'/uploads/{filename}',
            'processed_image_url': f'/uploads/{filename}',
            'filename': filename,
            'result': result
        })
        # Print the prediction for the image
        print(f"Prediction for {image_path}: {result}")
    
    # Debugging: Print the number of processed images
    print(f"Expected number of images: {len(image_paths)}, Processed images: {len(processed_images)}")
    
    return jsonify(processed_images)

@app.route('/stop_process', methods=['POST'])
def stop_process():
    global stop_processing
    stop_processing = True
    return jsonify({'status': 'Processing stopped'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(processed_output_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
