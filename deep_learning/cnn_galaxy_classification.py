# Goal: Classify Galaxy (Ellipticals, Spirals or Irregulars) using Pytorch looking the shape: galaxy-zoo-2-images-Kaggle

import torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchinfo import summary
from PIL import Image
import os
import pandas as pd
import logging
import sys

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class GalaxyDataset:
    def __init__(self):
        self.img_dir = 'C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\deep_learning\\cnn_galaxy\\images'
        self.mapping_galaxy_dir = pd.read_csv('C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\deep_learning\\cnn_galaxy\\gz2_filename_mapping.csv')
        self.df_scientific_dir = pd.read_csv("C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\deep_learning\\cnn_galaxy\\gz2_hart16.csv")

        self.data_complete = pd.merge(self.mapping_galaxy_dir, self.df_scientific_dir, left_on="objid",
                                      right_on="dr7objid")

        # Creates the class column (0, 1, 2) based on the morphological classification of the file (gz2_class)
        self.data_complete['target_class'] = -1
        self.data_complete.loc[
            self.data_complete['gz2_class'].str.startswith('E', na=False), 'target_class'] = 0  # Ellipticals
        self.data_complete.loc[
            self.data_complete['gz2_class'].str.startswith('S', na=False), 'target_class'] = 1  # Spirals
        self.data_complete.loc[
            self.data_complete['gz2_class'].str.startswith('I', na=False), 'target_class'] = 2  # Irregular

        # Keep just 3 categories
        self.data_complete = self.data_complete[self.data_complete['target_class'] != -1].reset_index(drop=True)
        self.img_names = self.data_complete['asset_id'].values
        self.labels = self.data_complete['target_class'].values

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        logger.info('Getting images')
        img_path = os.path.join(self.img_dir, f"{self.img_names[index]}.jpg")

        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class GalaxyPipeline:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_dataset, self.test_dataset = [None] * 2
        self.cnn_model, self.optimizer, self.criterion = [None] * 3

    def divide_train_test(self):
        logger.info('Dividing train and test with random split')

        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        self.train_dataset = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_dataset = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def model_cnn_galaxy(self, learning_rate=0.001):
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),  # Features maps needs to be equal of out_channels
            nn.ELU(),  # Works better with space (black plane)
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce the image to 112x112
            nn.Dropout2d(p=0.25),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce the size to 56x56
            nn.Dropout2d(p=0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce the size to 28x28
            nn.Dropout2d(p=0.25),

            nn.Flatten(),  # Single line vector

            nn.Linear(64 * 28 * 28, 128),  # final channels (64) * final width (28) * final height (28)
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 3)  # Output: 3 LOGITS
        )
        summary(self.cnn_model, input_size=(1, 3, 224, 224), device='cpu')
        self.optimizer = optim.Adam(self.cnn_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def running_model(self, num_epochs=10, patience=3, min_delta=0.001):
        logger.info('Run Model in Epochs')

        # Variables of Early Stopping
        best_loss = float('inf')  # It starts with infinity so that any loss is smaller.
        epochs_no_improve = 0
        best_model_weights = None

        for epoch in range(num_epochs):
            self.cnn_model.train()  # My model
            running_train_loss = 0.0

            for inputs, labels in self.train_dataset:
                self.optimizer.zero_grad(set_to_none=True)  # Clean Gradients of optimizer
                outputs = self.cnn_model(inputs)  # Forward: predictions

                loss = self.criterion(outputs, labels)  # CrossEntropyLoss
                loss.backward()
                self.optimizer.step()  # Update weights
                running_train_loss += loss.item()  # Accumulate the loss for monitoring purposes.

            epoch_loss = running_train_loss / len(self.train_dataset)
            logger.debug(f'Epoch [{epoch + 1}/{num_epochs}], Loss train: {epoch_loss:.4f}')

            # Validation/Test
            self.cnn_model.eval()  # Disable dropout and batchNorm
            running_test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():  # Disable gradients, won't use backward and optimizer
                for inputs, labels in self.test_dataset:
                    outputs = self.cnn_model(inputs)

                    # Loss of test for Early Stopping
                    loss = self.criterion(outputs, labels)
                    running_test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)  # Choose the class
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            epoch_test_loss = running_test_loss / len(self.test_dataset)
            accuracy = 100 * correct / total
            logger.debug(f'Epoch [{epoch + 1}/{num_epochs}] - Accuracy Test: {accuracy:.2f}%')

            # Early stopp working
            if epoch_test_loss < (best_loss - min_delta):
                best_loss = epoch_test_loss
                epochs_no_improve = 0

                best_model_weights = self.cnn_model.state_dict().copy()
                logger.info(f"==> New best loss weight, saving")
            else:
                epochs_no_improve += 1
                logger.info(f"==> Without improve {epochs_no_improve} Epoch(s).")

            if epochs_no_improve >= patience:
                logger.warning(f'Early Stopping triggered! Training stopped at that time {epoch + 1}.')
                break

        # Restores best weight
        if best_model_weights is not None:
            self.cnn_model.load_state_dict(best_model_weights)
            logger.info("Best weight restored.")

    def evaluate_model(self, device='cpu'):
        logger.info('Starting model evaluation and metrics plotting')

        # Put the model into evaluation mode and move it to the correct device.
        self.cnn_model.eval()
        self.cnn_model.to(device)

        all_predicts = []
        all_labels = []

        class_names = ['Ellipticals', 'Spirals', 'Irregular']

        with torch.no_grad():
            for inputs, labels in self.test_dataset:
                inputs = inputs.to(device)

                outputs = self.cnn_model(inputs)

                _, predicted = torch.max(outputs, 1)

                all_predicts.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        logger.info("CLASSIFICATION REPORT - GALAXY MORPHOLOGY")
        logger.info(classification_report(all_labels, all_predicts, target_names=class_names))

        cm = confusion_matrix(all_labels, all_predicts)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.title('Confusion Matrix - Galaxy Morphology', fontsize=14, pad=15)
        plt.ylabel('Real Classes', fontsize=12)
        plt.xlabel('Predict Classes', fontsize=12)
        plt.tight_layout()
        plt.show()






class_cnn = ConvolutionalNeuralNetworkGalaxy()
class_cnn.getting_data()
class_cnn.train_and_fit_model()
class_cnn.evaluate_model()
class_cnn.single_image_test()