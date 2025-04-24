# ========================== CNN Classifier =========================
import shared.models.model as m
import tensorflow as tf
from tensorflow.keras import layers, models

class cnn(m.Model):
    def __init__(self, x_train: list, y_train: list, x_test: list, y_test: list, dataset: str, seed: int) -> None:
        super().__init__(x_train, y_train, x_test, y_test, dataset, seed=seed)
        self.exe()

    def expecific_model(self) -> object:
        """Model CNN"""
        input_shape = (self.x_train.shape[1], 1)  # Adjust input shape for CNN
        num_classes = len(set(self.y_train))  # Calculate the number of classes

        # Create CNN model
        model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),

            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')  # Softmax for multi-class, sigmoid for binary
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',  # Use correct loss function
            metrics=['accuracy']
        )

        # Train the model
        model.fit(self.x_train[..., None], self.y_train, batch_size=32, epochs=1, verbose=1, validation_split=0.1)

        return model

    def __str__(self) -> str:
        """Returns the name of the class"""
        return "cnn"
