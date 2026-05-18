# Goal: Classify facial emotions by category
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
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
        self.train_dir = 'deep_learning\\train_cnn'
        self.test_dir = 'deep_learning\\test_cnn'
        self.single_image = 'deep_learning\\smile_fer.JPEG'

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
        #     layers.RandomZoom(0.1),])  # Move your face slightly closer to or further away from your face

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
        logger.info('Processing single image to predict')

        img = tf.io.read_file(self.single_image)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)  # Convert to grey (channels=1)

        img = tf.image.resize(img, self.img_size)  # Resize image to fit the template (48x48)
        img_array = tf.expand_dims(img, 0)  # [48, 48, 1] to [1, 48, 48, 1])

        predictions = self.model_convolutional.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])  # Higher probability
        confidence = predictions[0][predicted_class_idx] * 100

        class_names = self.test_dataset.class_names if self.test_dataset else ['angry', 'disgust', 'fear', 'happy',
                                                                               'neutral', 'sad', 'surprise']
        predicted_emotion = class_names[predicted_class_idx]

        logger.info("\n=== Results ===")
        logger.info(f"Emotions detected: {predicted_emotion.upper()}")
        logger.info(f"Confidence: {confidence:.2f}%")

        for name, prob in zip(class_names, predictions[0]):
            logger.info(f"-> {name}: {prob * 100:.2f}%")

        plt.figure(figsize=(6, 6))

        img_original_raw = tf.io.read_file(self.single_image)
        img_original = tf.image.decode_image(img_original_raw, expand_animations=False)

        if img_original.shape[-1] == 1:
            plt.imshow(tf.squeeze(img_original), cmap='gray')
        else:
            plt.imshow(img_original)

        plt.title(f"Emotion: {predicted_emotion.upper()} ({confidence:.1f}%)", fontsize=14, pad=10)
        plt.axis('off')
        plt.show()

        return predicted_emotion, confidence


class_cnn = ConvolutionalNeuralNetwork()
class_cnn.getting_data()
class_cnn.train_and_fit_model()
class_cnn.evaluate_model()
class_cnn.single_image_test()