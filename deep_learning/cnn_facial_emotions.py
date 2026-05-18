# Goal: Classify facial emotions by category
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys

# Logger setting
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Console will show everything

# Handler to console
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.train_dir = 'C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\deep_learning\\train_cnn'
        self.test_dir = 'C:\\Users\\Weslei\\Desktop\\Assuntos_de_estudo\\Assuntos_de_estudo\\Fases da vida\\Fase I\\Repository Projects\\files\\deep_learning\\test_cnn'

        # Image Settings of FER-2013 and batch size for my config
        self.batch_size = 64
        self.img_size = (48, 48)

        self.train_dataset, self.test_dataset = [None] * 2

        self.model_convolutional = None

    def getting_data(self):
        logger.info('Getting train and test data in files')

        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.train_dir,
            labels='inferred',  # Deduce the names of the classes, because we have many categories
            label_mode='categorical',  # Used for multiclass classification
            color_mode='grayscale',  # The FER-2013 is in shades of gray (1 channel).
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=True
        )

        self.test_dataset = tf.keras.utils.image_dataset_from_directory(
            self.test_dir,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=False  # Do not scramble the test to properly evaluate metrics
        )

    @staticmethod
    def model_cnn(learning_rate=0.001):

        # data_augmentation = models.Sequential([  # Adjusting problem of overfitting, won't run in test/validation
        #     layers.RandomFlip("horizontal"),  # Faces can look to the left or right.
        #     layers.RandomRotation(0.1),  # Slight head tilts (up to 10%)
        #     layers.RandomZoom(0.1),  # Move your face slightly closer to or further away from your face.
        # ])

        model = models.Sequential([
            # data_augmentation,  # It doesn't work on my PC; it needs more loading capacity.

            layers.Rescaling(1. / 255, input_shape=(48, 48, 1)),  # Normalize data [0, 255] to [0, 1]

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # Spatial Invariance
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),  # Size 48x48 to 24x24
            layers.Dropout(0.25),  # DROPOUT 1: Disconnect 25% of connections after the first pooling

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),  # Reduce size again
            layers.Dropout(0.30),  # Avoid overfitting

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # Capturing complex expressions (12x12)
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),  # 6x6
            layers.Dropout(0.40),

            layers.Flatten(),  # Adjust matrix in a linear vector
            layers.Dense(256, activation='relu'),  # Deep layers
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # DROPOUT 4: It shuts down 50% of dense neurons to prevent rote memorization

            layers.Dense(7, activation='softmax')  # Softmax because have 7 class of emotions
        ])

        model.build(input_shape=(None, 48, 48, 1))
        model.summary()

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_and_fit_model(self):
        logger.info('Fit and create a model')

        self.model_convolutional = self.model_cnn()

        es = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    
        # It halves the learning rate (factor=0.5) if the loss does not decrease for 2 epochs.
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )

        # Labels of classes
        y_train_labels = np.concatenate([y for x, y in self.train_dataset], axis=0).argmax(axis=1)
        unique_class = np.unique(y_train_labels)
        weight_class = compute_class_weight('balanced', classes=unique_class, y=y_train_labels)
        dict_weight_class = dict(zip(unique_class, weight_class))

        history = self.model_convolutional.fit(
            self.train_dataset,
            validation_data=self.test_dataset,
            epochs=60,  # Early stop will work before complete 60
            class_weight=dict_weight_class,
            callbacks=[es, reduce_lr]
        )
        return history

    def evaluate_model(self):
        logger.info('Evaluating model')
        y_true = np.concatenate([y for x, y in self.test_dataset], axis=0).argmax(axis=1)

        y_predict_probs = self.model_convolutional.predict(self.test_dataset)
        y_predict = y_predict_probs.argmax(axis=1)

        logger.info("\n--- Classification Report ---")
        logger.info(classification_report(y_true, y_predict, target_names=self.test_dataset.class_names))

        cm = confusion_matrix(y_true, y_predict)
        class_names = self.test_dataset.class_names
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.title('Confusion Matrix - Facial emotions', fontsize=14, pad=15)
        plt.ylabel('Real Classes', fontsize=12)
        plt.xlabel('Predict Classes', fontsize=12)
        plt.tight_layout()
        plt.show()

    def single_image_test(self):
        # Matplotlib para mostrar a foto e o resultado
        pass


class_cnn = ConvolutionalNeuralNetwork()
class_cnn.getting_data()
class_cnn.train_and_fit_model()
class_cnn.evaluate_model()