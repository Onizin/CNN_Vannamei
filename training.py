import numpy as np
import torch
import pandas as pd
import psutil
from matplotlib import pyplot as plt
from activation import Activation
from Dense import init_params, update_params, one_hot, update_params_adam, update_params_rmsprop

def normalize_data(X):
    if X.ndim == 1:
        mean = np.mean(X)
        std = np.std(X)
    else:
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
    
    # Avoid division by zero
    std[std == 0] = 1
    return (X - mean) / std

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = foward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def foward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1  # Perkalian dot antara input dengan weight + bias 
    A1 = Activation.Relu(Z1)  # Aktivasi relu
    Z2 = W2.dot(A1) + b2  # Perkalian dot antara hidden dengan weight + bias
    A2 = Activation.Softmax(Z2)  # Aktivasi softmax
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * Activation.ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def compute_loss(A2, Y, W1, W2, lambda_):
    m = Y.shape[0]
    logprobs = -np.log(A2[Y, range(m)])
    loss = np.sum(logprobs) / m
    l2_regularization = (lambda_ / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    loss += l2_regularization
    return loss

def gradient_descent(X, Y, alpha, iterations, memory_threshold=50, save_interval=10, target_loss=0.01, epochs=10, optimizer="gradient_descent"):
    input_size = X.shape[0]
    hidden_size = 4  # You can adjust this value as needed
    output_size = len(np.unique(Y))
    
    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)
    W1_history, b1_history, W2_history, b2_history = [], [], [], []
    accuracies = []
    losses = []
    best_accuracy = 0
    best_params = (W1, b1, W2, b2)

    # Initialize Adam and RMSProp variables
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8 # Hyperparameters for Adam and RMSProp
    mW1, mb1, mW2, mb2 = 0, 0, 0, 0 # Initialize first moment vectors for Adam
    vW1, vb1, vW2, vb2 = 0, 0, 0, 0 # Initialize second moment vectors for Adam and RMSProp

    for epoch in range(epochs):
        for i in range(iterations):
            Z1, A1, Z2, A2 = foward_prop(W1, b1, W2, b2, X)

            # Compute loss
            loss = compute_loss(A2, Y, W1, W2, lambda_=0.01)
            loss = round(loss, 4)
            losses.append(loss)

            if loss <= target_loss:
                print(f"Target loss reached at epoch {epoch}, iteration {i}")
                break

            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

            if optimizer == "gradient_descent":
                W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            elif optimizer == "adam":
                W1, b1, mW1, mb1, vW1, vb1 = update_params_adam(W1, b1, dW1, db1, mW1, mb1, vW1, vb1, beta1, beta2, alpha, epsilon, i + 1)
                W2, b2, mW2, mb2, vW2, vb2 = update_params_adam(W2, b2, dW2, db2, mW2, mb2, vW2, vb2, beta1, beta2, alpha, epsilon, i + 1)

            elif optimizer == "rmsprop":
                W1, b1, vW1, vb1 = update_params_rmsprop(W1, b1, dW1, db1, vW1, vb1, beta2, alpha, epsilon)
                W2, b2, vW2, vb2 = update_params_rmsprop(W2, b2, dW2, db2, vW2, vb2, beta2, alpha, epsilon)

            if i % save_interval == 0:
                W1_history.append(W1.copy())
                b1_history.append(b1.copy())
                W2_history.append(W2.copy())
                b2_history.append(b2.copy())

            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss}")
                predictions = get_predictions(A2)
                accuracy = get_accuracy(predictions, Y)
                accuracy = round(accuracy, 4)
                print("Accuracy: ", accuracy)
                
                # Always append accuracy for plotting (not just every 100 iterations)
                accuracies.append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

                # Stop training if accuracy reaches 100%
                if accuracy == 1.0:
                    print(f"100% accuracy reached at epoch {epoch}, iteration {i}. Stopping training.")
                    # Save plots and parameters before returning
                    file_trainsave = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\z_train\\train.pt'
                    torch.save({
                        'W1': best_params[0],
                        'b1': best_params[1],
                        'W2': best_params[2],
                        'b2': best_params[3]
                    }, file_trainsave)
                    plot_and_save_loss(losses, save_path='C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\z_train\\confusion_matrix\\loss.png')
                    plot_and_save_training_accuracy(accuracies, save_path='C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\z_train\\confusion_matrix\\train_accu.png')
                    return best_params, W1_history, b1_history, W2_history, b2_history

            if i % 100 == 0:
                # Remove the duplicate accuracies.append since it's now done every 10 iterations
                pass

            memory_usage = psutil.virtual_memory().percent
            if memory_usage > memory_threshold:
                print(f"Memory usage is above {memory_threshold}%. Optimizing memory usage.")

        print(f"Epoch {epoch} completed. Best Accuracy so far: {best_accuracy}")

    print("Best Accuracy: ", best_accuracy)

    # Save best parameters
    file_trainsave = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\z_train\\train.pt'
    torch.save({
        'W1': best_params[0],
        'b1': best_params[1],
        'W2': best_params[2],
        'b2': best_params[3]
    }, file_trainsave)
    plot_and_save_loss(losses, save_path='C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\z_train\\confusion_matrix\\loss.png')
    plot_and_save_training_accuracy(accuracies, save_path='C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\z_train\\confusion_matrix\\train_accu.png')
    return best_params, W1_history, b1_history, W2_history, b2_history

def plot_and_save_training_accuracy(accuracies, interval=10, save_path=None):
    plt.figure(figsize=(15, 6))
    plt.plot(range(0, len(accuracies) * interval, interval), accuracies, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Iterations')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_and_save_loss(losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def test_prediction(index, X, Y, W1, b1, W2, b2):
    current_image = X[:, index]
    expected_size = W1.shape[1]
    if current_image.size != expected_size:
        print(f"Current image size is not {expected_size}, cannot reshape to ({expected_size}, 1)")
        return None
    current_image = current_image.reshape((expected_size, 1))  # Adjust reshaping to match expected input shape
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y[index]
    return prediction[0], label

def plot_and_save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')

    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Label')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.title('ConfusionMatrix')
    plt.savefig(save_path)
    plt.show()

def calculate_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def custom_train_test_split(X, y, filenames, test_size=0.25, random_state=None, stratify=None):
    if random_state is not None:
        np.random.seed(random_state)
    # Acak data
    indices = np.arange(X.shape[0])
    if stratify is not None:
        # Stratified shuffle split
        unique_classes, class_counts = np.unique(stratify, return_counts=True)
        test_indices = []
        train_indices = []
        for cls in unique_classes:
            cls_indices = indices[stratify == cls]
            np.random.shuffle(cls_indices)
            split_point = int(len(cls_indices) * (1 - test_size))
            train_indices.extend(cls_indices[:split_point])
            test_indices.extend(cls_indices[split_point:])
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
    else:
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_size))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    filenames_train = np.array(filenames)[train_indices]
    filenames_test = np.array(filenames)[test_indices]

    # Save train data to file
    # train_data_path = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/after_process/train_data.txt'
    train_data_path = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\after_process\\train_data.txt'
    with open(train_data_path, 'w') as train_file:
        for idx, label, filename in zip(train_indices, y_train, filenames_train):
            train_file.write(f"Index: {idx}, Label: {label}, Filename: {filename}\n")

    # Save test data to file
    # test_data_path = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/after_process/test_data.txt'
    test_data_path = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\after_process\\test_data.txt'
    with open(test_data_path, 'w') as test_file:
        for idx, label, filename in zip(test_indices, y_test, filenames_test):
            test_file.write(f"Index: {idx}, Label: {label}, Filename: {filename}\n")
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices, filenames_train, filenames_test

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def calculate_precision_recall_f1(cm):
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

def plot_metrics(precision, recall, f1_score, class_names, save_path):
    x = np.arange(len(class_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1_score, width, label='F1 Score')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1 Score by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def save_test_prediction_accuracy(predictions, true_labels, save_path, class_names, test_indices, filenames_test):
    accuracy = calculate_accuracy(true_labels, predictions)
    num_classes = len(class_names)
    cm = calculate_confusion_matrix(true_labels, predictions, num_classes)
    precision, recall, f1_score = calculate_precision_recall_f1(cm)

    with open(save_path, 'w') as f:
        f.write(f"Test Prediction Accuracy: {accuracy * 100:.2f}%\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\nPrecision:\n")
        f.write(np.array2string(precision))
        f.write("\nRecall:\n")
        f.write(np.array2string(recall))
        f.write("\nF1 Score:\n")
        f.write(np.array2string(f1_score))

    print(f"Test Prediction Accuracy and metrics saved to {save_path}")

    # Save TP, TN, FP, FN results
    tp_path = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\after_process\\TP.txt'
    tn_path = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\after_process\\TN.txt'
    fp_path = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\after_process\\FP.txt'
    fn_path = 'C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\after_process\\FN.txt'

    # tp_path = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/after_process/TP.txt'
    # tn_path = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/after_process/TN.txt'
    # fp_path = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/after_process/FP.txt'
    # fn_path = '/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/after_process/FN.txt'
    with open(tp_path, 'w') as tp_file, open(tn_path, 'w') as tn_file, open(fp_path, 'w') as fp_file, open(fn_path, 'w') as fn_file:
        for idx, (true_label, pred_label, filename) in enumerate(zip(true_labels, predictions, filenames_test)):
            if true_label == pred_label:
                if true_label == 1:  # True Positive
                    tn_file.write(f"Index: {test_indices[idx]}, True Label: {true_label}, Predicted Label: {pred_label}, Filename: {filename}\n")
                else:  # True Negative
                    tp_file.write(f"Index: {test_indices[idx]}, True Label: {true_label}, Predicted Label: {pred_label}, Filename: {filename}\n")
            else:
                if pred_label == 1:  # False Positive
                    fp_file.write(f"Index: {test_indices[idx]}, True Label: {true_label}, Predicted Label: {pred_label}, Filename: {filename}\n")
                else:  # False Negative
                    fn_file.write(f"Index: {test_indices[idx]}, True Label: {true_label}, Predicted Label: {pred_label}, Filename: {filename}\n")

    plot_and_save_confusion_matrix(cm, class_names, "C:\\ujicobaprogram\\ujicoba MLP\\feature_Extract\\z_train\\confusion_matrix")
    # plot_and_save_confusion_matrix(cm, class_names,
    #                                "/media/ai/Micro/Coding/vannamei/ujicoba MLP/feature_Extract/z_train/confusion_matrix/")

    plot_metrics(precision, recall, f1_score, class_names, save_path.replace('.txt', '_metrics.png'))
