import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt 
from RGBtoGray import rgb2gray

def load_dataset(dataset_path):
    data = []
    labels = []
    filenames = []
    # Membaca dataset dari folder
    for label in ["Udang_Vannamei", "Bukan_Udang"]:  # Nama folder adalah label
        folder_path = os.path.join(dataset_path, label)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):  # Hanya membaca file gambar
                # Membaca dan memproses gambar
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("L")  # Membaca gambar dan konversi ke grayscale
                img_array = image_to_array(img)  # Konversi gambar ke array
                img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi channel
                data.append(img_array)  # Menambahkan gambar yang diproses
                labels.append(0 if label == "Udang_Vannamei" else 1)  # 0: udang Vannamei, 1: Bukan udang
                filenames.append(filename)  # Menyimpan nama file

    # Simpan nama file ke dalam file txt
    output_dir = "c:\\ujicobaprogram\\Deteksi Vannamei\\feature_Extract\\after_process\\"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "index.txt"), "w") as f:
        for idx, filename in enumerate(filenames):
            f.write(f"{idx}\t{filename}\n")

    return data, labels, filenames

def random_initialize_data(data, labels, filenames):
    indices = torch.randperm(len(data))  # Menghasilkan indeks acak
    data = torch.round(torch.tensor(data, dtype=torch.uint8))[indices]
    labels = torch.tensor(labels)[indices].clone().detach().long()
    shuffled_filenames = [filenames[i] for i in indices]

    # Simpan nama file yang diacak ke dalam file txt
    output_dir = "c:\\ujicobaprogram\\Deteksi Vannamei\\feature_Extract\\after_process\\"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "shuffle_index.txt"), "w") as f:
        for idx, filename in enumerate(shuffled_filenames):
            f.write(f"{idx}\t{filename}\n")

    return data, labels, shuffled_filenames

def resize_image(image, size=(224, 224)):
    return image.resize(size)

def image_to_array(image):
    return np.array(image)  # Mengonversi gambar ke array NumPy

def prossesgray_save(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    label_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for label_folder in label_folders:
        label = os.path.basename(label_folder)
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)
        
        image_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_path in image_paths:
            with Image.open(image_path) as img:
                img_resized = resize_image(img)  # Resize gambar
                gray_image = rgb2gray(np.array(img_resized))
                gray_image_pil = Image.fromarray(gray_image).convert("L")  # Convert to mode "L" for grayscale
                
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(output_label_dir, f"{base_filename}_gray.jpg")  # Save as JPG
                gray_image_pil.save(save_path)
                print(f"Processed and saved image to {save_path}")

# menampilkan hasil gambar
def display_samples(data1, title1, data2, title2, cmap='gray'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(data1, cmap=cmap)
    axes[0].set_title(title1)
    axes[0].axis("off")
    
    axes[1].imshow(data2, cmap=cmap)
    axes[1].set_title(title2)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

def display_imagesRGB(data, labels, num_samples):
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 3))
    for i in range(num_samples):
        axes[i].imshow(data[i].squeeze(0))  # Menampilkan gambar RGB
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")  # Menghilangkan sumbu
    plt.show()

def display_imagesGray(data, labels, num_samples):
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 5))
    for i in range(num_samples):
        pooled_image = data[i]  # Remove .detach().numpy() to keep it as a tensor
        if pooled_image.ndimension() == 3:
            pooled_image = pooled_image.squeeze(0)  # Remove channel dimension if present
        axes[i].imshow(pooled_image, cmap='gray')  # Tampilkan gambar
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")  # Sembunyikan sumbu
    plt.show()

def change_color_to_rgb(image, markers, labels):
    # Create a mask for the labels
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for label in labels[2:]:
        mask[markers == label] = 255
    
    # Change the color of the non-label areas to black
    image[mask == 0] = [0, 0, 0]
    return image

def change_color_to_rgb(image, markers, labels):
    # Create a mask for the labels
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for label in labels[2:]:
        mask[markers == label] = 255
    
    # Change the color of the non-label areas to black
    image[mask == 0] = [0, 0, 0]
    return image

def process_and_save_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    label_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for label_folder in label_folders:
        label = os.path.basename(label_folder)
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)
        
        image_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for idx, image_path in enumerate(image_paths):
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
            dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
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
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_label_dir, f"{base_filename}_removal.jpg")
            cv2.imwrite(output_path, img)
            print(f"Saved: {output_path}")

def merge_data(input_dirs, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for input_dir in input_dirs:
        label_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        
        for label_folder in label_folders:
            label = os.path.basename(label_folder)
            output_label_dir = os.path.join(output_dir, label)
            if not os.path.exists(output_label_dir):
                os.makedirs(output_label_dir)
            
            image_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png') or f.endswith('.jpg')]
            
            for image_path in image_paths:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error: Gambar tidak ditemukan atau tidak dapat dibaca di path: {image_path}")
                    continue
                
                base_filename = os.path.basename(image_path)
                output_path = os.path.join(output_label_dir, base_filename)
                cv2.imwrite(output_path, img)
                print(f"Saved: {output_path}")