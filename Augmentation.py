import cv2
import numpy as np
import os

def augment_rotation_flipping(input_dir, output_dir):
    # Membuat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mendapatkan semua subfolder dalam direktori input
    label_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for label_folder in label_folders:
        label = os.path.basename(label_folder)
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        # Mendapatkan semua path gambar dalam subfolder label
        image_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png') or f.endswith('.jpg')]

        for image_path in image_paths:
            # Membaca gambar
            image = cv2.imread(image_path)  # Membaca gambar dalam format warna (RGB)
            if image is None:
                print(f"Error: Gambar tidak ditemukan atau tidak dapat dibaca di path: {image_path}")
                continue

            # List untuk menyimpan gambar hasil augmentasi
            augmented_images = []

            # Original image
            augmented_images.append(('original', image))

            # Mendapatkan ukuran gambar asli
            h, w = image.shape[:2]

            # Membuat kanvas yang lebih besar untuk menampung gambar yang diputar
            diagonal = int(np.sqrt(h**2 + w**2))
            canvas_size = (diagonal, diagonal, 3)
            canvas_center = (diagonal // 2, diagonal // 2)

            # Warna background (asumsi warna background adalah warna piksel di sudut kiri atas)
            background_color = image[0, 0].tolist()

            # Rotasi dengan interval 10 derajat dari 0 hingga 270 derajat
            for angle in range(0, 271, 10):
                # Membuat kanvas kosong dengan warna background
                canvas = np.full(canvas_size, background_color, dtype=image.dtype)
                
                # Menempatkan gambar asli di tengah kanvas
                x_offset = (diagonal - w) // 2
                y_offset = (diagonal - h) // 2
                canvas[y_offset:y_offset + h, x_offset:x_offset + w] = image

                # Mendapatkan matriks rotasi
                rotation_matrix = cv2.getRotationMatrix2D(canvas_center, angle, 1.0)
                rotated_image = cv2.warpAffine(canvas, rotation_matrix, (diagonal, diagonal))
                
                # Memotong gambar yang diputar agar sesuai dengan ukuran asli
                cropped_image = rotated_image[y_offset:y_offset + h, x_offset:x_offset + w]
                augmented_images.append((f'rotated_{angle}', cropped_image))

            # Flipping horizontal
            flipped_h = cv2.flip(image, 1)
            augmented_images.append(('flipped_h', flipped_h))

            # Flipping vertical
            flipped_v = cv2.flip(image, 0)
            augmented_images.append(('flipped_v', flipped_v))

            # Menyimpan gambar hasil augmentasi
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            for name, img in augmented_images:
                output_path = os.path.join(output_label_dir, f"{base_filename}_{name}.jpg")
                cv2.imwrite(output_path, img)
                print(f"Saved: {output_path}")

def augment_noise(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mendapatkan semua subfolder dalam direktori input
    label_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for label_folder in label_folders:
        label = os.path.basename(label_folder)
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        # Mendapatkan semua path gambar dalam subfolder label
        image_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png') or f.endswith('.jpg')]

        for image_path in image_paths:
            # Membaca gambar
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Gambar tidak ditemukan atau tidak dapat dibaca di path: {image_path}")
                continue

            # Menghilangkan noise menggunakan teknik denoising
            denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

            # Menyimpan gambar hasil augmentasi
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_label_dir, f"{base_filename}_denoised.jpg")
            cv2.imwrite(output_path, denoised_image)
            print(f"Saved: {output_path}")

def augment_contrast(input_dir, output_dir, alpha_range=(1.0, 2.0), alpha_step=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mendapatkan semua subfolder dalam direktori input
    label_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for label_folder in label_folders:
        label = os.path.basename(label_folder)
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        # Mendapatkan semua path gambar dalam subfolder label
        image_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png') or f.endswith('.jpg')]

        for image_path in image_paths:
            # Membaca gambar
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Gambar tidak ditemukan atau tidak dapat dibaca di path: {image_path}")
                continue

            # Mengubah kontras dengan berbagai nilai alpha
            for alpha in np.arange(alpha_range[0], alpha_range[1], alpha_step):
                contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

                # Menyimpan gambar hasil augmentasi
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_label_dir, f"{base_filename}_contrast_{alpha:.1f}.jpg")
                cv2.imwrite(output_path, contrast_image)
                print(f"Saved: {output_path}")

def augment_brightness(input_dir, output_dir, beta_range=(0, 100), beta_step=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Mendapatkan semua subfolder dalam direktori input
    label_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for label_folder in label_folders:
        label = os.path.basename(label_folder)
        output_label_dir = os.path.join(output_dir, label)
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)

        # Mendapatkan semua path gambar dalam subfolder label
        image_paths = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.png') or f.endswith('.jpg')]

        for image_path in image_paths:
            # Membaca gambar
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Gambar tidak ditemukan atau tidak dapat dibaca di path: {image_path}")
                continue

            # Mengubah kecerahan dengan berbagai nilai beta
            for beta in range(beta_range[0], beta_range[1] + 1, beta_step):
                brightness_image = cv2.convertScaleAbs(image, alpha=1, beta=beta)

                # Menyimpan gambar hasil augmentasi
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_label_dir, f"{base_filename}_brightness_{beta}.jpg")
                cv2.imwrite(output_path, brightness_image)
                print(f"Saved: {output_path}")

