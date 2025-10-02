import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from Convolution import Conv2D, pool2d, flatten
from preprocessing import load_dataset, random_initialize_data, display_samples, prossesgray_save, display_imagesRGB, display_imagesGray
from training import gradient_descent, test_prediction, normalize_data, custom_train_test_split
from training import calculate_accuracy, plot_and_save_confusion_matrix
from sklearn.metrics import confusion_matrix
from training import save_test_prediction_accuracy

dataset_path = "C:\\CNN_Vannamei\\gambarinput\\preprocessing\\resize_grayscale"

data, labels, filenames = load_dataset(dataset_path)

data, labels, filenames = random_initialize_data(data, labels, filenames)
# print(data.shape)
num_samples = 2

#menampilkan gambar
# display_imagesGray(data, labels, num_samples)

# kernel atau filter
K = torch.tensor([
    [[[-2.0, -1.0, 0.0],
      [-1.0, 1.0, 1.0],
      [0.0, 1.0, 2.0]]],
    
    [[[-1.0, -1.0, -1.0],
      [-1.0, 8.0, -1.0],
      [-1.0, -1.0, -1.0]]]
], dtype=torch.float32)
Kernel = K

# # print(Kernel.shape)

# ==================================================================
# layer 1 conv2d
conv2d = Conv2D(Kernel,padding=1)
Conv = conv2d(data.float())
print(Conv.shape)
output_fileconv1 = 'c:\\CNN_vannamei\\feature_Extract\\conv1\\conv1.pt'
data_to_save = {
    'conv': Conv,
    'labels': labels,
    'filenames': filenames
}
torch.save(data_to_save, output_fileconv1)

# display_imagesGray(Conv, labels, num_samples)

# print(Conv.shape)
# # tampilkan perbedaan sebelum di konvolusi dan sesudah
# # sampel_index = 0
# # # data awal sebelum di konvolusi
# # sample_data = segmen[sampel_index, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama
# # sample_data_10x10 = sample_data[:10, :10]  # Ambil 10x10 sampel

# # # dikonvolusi
# # sample_conv_tensor = torch.tensor(sample_data_10x10).unsqueeze(0).unsqueeze(0)  # Tambahkan batch dan channel dimension
# # Conv_sample = conv2d(sample_conv_tensor.float())  # Pastikan data adalah float

# # # Ambil hasil konvolusi dari sampel 10x10
# # sampleconv_img_10x10 = Conv_sample[0, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama

# # display_samples(sample_data_10x10, f"10x10 Sampel dari Gambar {sampel_index} sebelum Konvolusi Layer 1",
# #                 sampleconv_img_10x10, f"10x10 Sampel dari Gambar {sampel_index} setelah Konvolusi Layer 1")
# # ==================================================================

# # # ==================================================================
# pooling layer 1
Pool = pool2d(Conv,(2,2))
print(Pool.shape)
# display_images(Pool, labels, num_samples)
output_filepool1 = 'C:\\CNN_Vannamei\\feature_Extract\\pool1\\pool1.pt'
data_to_save = {
    'pool': Pool,
    'labels': labels,
    'filenames': filenames
}
torch.save(data_to_save, output_filepool1)

# # data awal sebelum pooling
# sample_data_p = Conv[sampel_index, 0].detach().cpu().numpy()  # Ambil channel pertama dari gambar pertama
# sample_datap_10x10 = sample_data_p[:10, :10]  # Ambil 10x10 sampel

# # data setelah pooling
# sample_pool_tensor = torch.tensor(sample_datap_10x10).unsqueeze(0).unsqueeze(0)  # Tambahkan batch dan channel dimension
# pool_sample = pool2d(sample_pool_tensor.float(),(2,2))  # Pastikan data adalah float

# # Ambil hasil pooling dari sampel 10x10
# samplepool_img_10x10 = pool_sample[0, 0].detach().numpy()

# display_samples(sample_datap_10x10, f"10x10 Sampel dari Gambar {sampel_index} sebelum Pooling layer 1",
#                 samplepool_img_10x10, f"10x10 Sampel dari Gambar {sampel_index} setelah Pooling layer 1")
# # ==================================================================

# # # ==================================================================
# layer 2 conv2d
Kernel2 = torch.tensor([
    [[[-2.0, -1.0, 0.0],
      [-1.0, 1.0, 1.0],
      [0.0, 1.0, 2.0]],
     [[-1.0, -1.0, -1.0],
      [-1.0, 8.0, -1.0],
      [-1.0, -1.0, -1.0]]],
    
    [[[-1.0, -1.0, -1.0],
      [-1.0, 8.0, -1.0],
      [-1.0, -1.0, -1.0]],
     [[-2.0, -1.0, 0.0],
      [-1.0, 1.0, 1.0],
      [0.0, 1.0, 2.0]]]
], dtype=torch.float32)
print(Kernel2.shape)
conv2d_2 = Conv2D(Kernel2, padding=1)
Conv_2 = conv2d_2(Pool.float())
print(Conv_2.shape)
output_fileconv2 = 'C:\\CNN_Vannamei\\feature_Extract\\conv2\\conv2.pt'
data_to_save = {
    'conv2': Conv_2,
    'labels': labels,
    'filenames': filenames
}
torch.save(data_to_save, output_fileconv2)

# display_imagesGray(Conv_2, labels, num_samples)

# # # # # tampilkan perbedaan sebelum di komvolusi dan sesudah
# # # # sample_data_p2 = Pool[sampel_index, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama
# # # # sample_datap2_10x10 = sample_data_p2[:10, :10]  # Ambil 10x10 sampel

# # # # # dikonvolusi
# # # # sample_conv2_tensor = torch.tensor(sample_datap2_10x10).unsqueeze(0).unsqueeze(0)  # Tambahkan batch dan channel dimension
# # # # Conv2_sample = conv2d(sample_conv2_tensor.float())  # Pastikan data adalah float

# # # # # Ambil hasil konvolusi dari sampel 10x10
# # # # sampleconv2_img_10x10 = Conv2_sample[0, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama

# # # # display_samples(sample_datap2_10x10 , f"10x10 Sampel dari Gambar {sampel_index} sebelum Konvolusi layer 2",
# # # #                 sampleconv2_img_10x10, f"10x10 Sampel dari Gambar {sampel_index} setelah Konvolusi layer 2")
# # # # ==================================================================

# # # ==================================================================
# # # pooling layer 2
Pool_2 = pool2d(Conv_2,(2,2))
print(Pool_2.shape)
output_filepool2 = 'C:\\CNN_Vannamei\\feature_Extract\\pool2\\pool2.pt'
data_to_save = {
    'pool2': Pool_2,
    'labels': labels,
    'filenames': filenames
}
torch.save(data_to_save, output_filepool2)

# # display_images(Pool_2, labels, num_samples)

# # # sample_data_c2 = Conv_2[sampel_index, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama
# # # sample_datac2_10x10 = sample_data_c2[:10, :10]  # Ambil 10x10 sampel

# # # # dikonvolusi
# # sample_pool2_tensor = torch.tensor(sample_datac2_10x10).unsqueeze(0).unsqueeze(0)  # Tambahkan batch dan channel dimension
# # Pool2_sample = pool2d(sample_pool2_tensor.float(),(2,2))  # Pastikan data adalah float

# # # # Ambil hasil konvolusi dari sampel 10x10
# # samplepool2_img_10x10 = Pool2_sample[0, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama

# # # # display_samples(sample_datac2_10x10, f"10x10 Sampel dari Gambar {sampel_index} sebelum Pooling layer 2",
# # # #                 samplepool2_img_10x10, f"10x10 Sampel dari Gambar {sampel_index} setelah Pooling layer 2")

# # # # ==================================================================

# # # # ==================================================================
# # layer 3 conv2d
Conv_3 = conv2d_2(Pool_2.float())
print(Conv_3.shape)

output_fileconv3 = 'C:\\CNN_Vannamei\\feature_Extract\\conv3\\conv3.pt'
data_to_save = {
    'conv3': Conv_3,
    'labels': labels,
    'filenames': filenames
}
torch.save(data_to_save, output_fileconv3)

# display_imagesGray(Conv_3, labels, num_samples)

# tampilkan perbedaan sebelum di komvolusi dan sesudah

# sample_data_pool2 = Pool_2[sampel_index, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama
# sample_datapool2_10x10 = sample_data_pool2[:10, :10]  # Ambil 10x10 sampel

# # dikonvolusi
# sample_conv3_tensor = torch.tensor(sample_datapool2_10x10).unsqueeze(0).unsqueeze(0)  # Tambahkan batch dan channel dimension
# Conv3_sample = conv2d(sample_conv3_tensor.float())  # Pastikan data adalah float

# # Ambil hasil konvolusi dari sampel 10x10
# sampleconv3_img_10x10 = Conv3_sample[0, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama

# display_samples(sample_datapool2_10x10, f"10x10 Sampel dari Gambar {sampel_index} sebelum Konvolusi layer 3",
#                 sampleconv3_img_10x10, f"10x10 Sampel dari Gambar {sampel_index} setelah Konvolusi layer 3")

# # ==================================================================

# # ==================================================================
# # pooling layer 3
Pool_3 = pool2d(Conv_3,(2,2))
print(Pool_3.shape)
output_filepool3 = 'C:\\CNN_Vannamei\\feature_Extract\\pool3\\pool3.pt'
data_to_save = {
    'pool3': Pool_3,
    'labels': labels,
    'filenames': filenames
}
torch.save(data_to_save, output_filepool3)

# display_imagesGray(Pool_3, labels, num_samples)

# sample_data_c3 = Conv_3[sampel_index, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama
# sample_datac3_10x10 = sample_data_c3[:10, :10]  # Ambil 10x10 sampel

# # dikonvolusi
# sample_pool3_tensor = torch.tensor(sample_datac3_10x10).unsqueeze(0).unsqueeze(0)  # Tambahkan batch dan channel dimension
# Pool3_sample = pool2d(sample_pool3_tensor.float(),(2,2))  # Pastikan data adalah float

# # # Ambil hasil konvolusi dari sampel 10x10
# samplepool3_img_10x10 = Pool3_sample[0, 0].detach().numpy()  # Ambil channel pertama dari gambar pertama

# display_samples(sample_datac3_10x10, f"10x10 Sampel dari Gambar {sampel_index} sebelum Pooling layer 3",
#                 samplepool3_img_10x10, f"10x10 Sampel dari Gambar {sampel_index} setelah Pooling layer 3")


# tahap training
# ==================================================================
# flatten
file_path = 'C:\\CNN_Vannamei\\feature_Extract\\pool3\\pool3.pt'
# # num_samples = 4  # Jumlah sampel yang akan ditampilkan
# # Memuat kembali tensor dan label dari file
loaded_data = torch.load(file_path)
loaded_conv = loaded_data['pool3']
loaded_labels = loaded_data['labels']
loaded_filenames = loaded_data['filenames']

# # Menampilkan informasi tensor dan label yang dimuat
print(f"Loaded conv tensor shape: {loaded_conv.shape}")
print(f"Loaded labels tensor shape: {loaded_labels.shape}")

# Flatten the conv tensor
F = flatten(loaded_conv)

# # Save the flattened tensor along with labels and filenames
flattened_output_file = 'C:\\CNN_Vannamei\\feature_Extract\\after_process\\flattened_output.pt'

data_to_save = {
    'flattened': F,
    'labels': loaded_labels,
    'filenames': loaded_filenames
}
torch.save(data_to_save, flattened_output_file)

print(F.shape)
print(F)
if F.is_cuda:  # Check if the tensor is on GPU
    F = F.cpu()
numpys = F.detach().numpy()


# Memuat kembali tensor dan label dari file
loaded_data = torch.load(flattened_output_file)
loaded_flat = loaded_data['flattened']
loaded_labels = loaded_data['labels']
loaded_filenames = loaded_data['filenames']


# Bagi data menjadi set pelatihan dan set pengujian
X_train, X_test, Y_train, Y_test, train_indices, test_indices, filenames_train, filenames_test = custom_train_test_split(loaded_flat, loaded_labels, loaded_filenames ,test_size=0.2, random_state=42, stratify=loaded_labels)

# Simpan indeks asli dari data uji
print(test_indices)
print(test_indices.shape)
# Cetak beberapa informasi tentang data
print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_test:", Y_test.shape)

# Normalize the data
X_train = normalize_data(X_train.detach().numpy().T)
Y_train = Y_train.detach().numpy()  # No need to normalize labels
X_test = normalize_data(X_test.detach().numpy().T)
Y_test = Y_test.detach().numpy()  # No need to normalize labels

learning_rate = 0.0015
iterations = 100
memory_threshold = 60  # Ambang batas penggunaan memori dalam persen
save_interval = 10  # Interval untuk menyimpan riwayat parameter
epochs = 10  # Number of epochs for training

best_params, W1_history, b1_history, W2_history, b2_history = gradient_descent(X_train, Y_train, learning_rate, iterations, memory_threshold, save_interval, epochs=epochs, optimizer="rmsprop")

# Visualisasi perubahan parameter
# plot_parameters(W1_history, b1_history, W2_history, b2_history, interval=10)
W1, b1, W2, b2 = best_params

# Melakukan prediksi pada data uji
predictions = []
true_labels = []
for idx in range(X_test.shape[1]):
    result = test_prediction(idx, X_test, Y_test, W1, b1, W2, b2)
    if result is not None:
        prediction, label = result
        predictions.append(prediction)
        true_labels.append(label)

predictions = np.array(predictions)
true_labels = np.array(true_labels)
print(predictions)
print(true_labels)

# Save test prediction accuracy and metrics
test_prediction_accuracy_path = 'C:\\CNN_Vannamei\\feature_Extract\\z_train\\confusion_matrix\\test_predict.txt'
save_test_prediction_accuracy(predictions, true_labels, test_prediction_accuracy_path, ['Udang_Vannamei', 'Bukan_Udang'], test_indices, filenames_test)