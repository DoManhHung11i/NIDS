# ========================== CNN Classifier =========================
#
#                   Author:  Sergio Arroni Del Riego (Modified)
#
# ===================================================================

# ==================> Imports
import shared.models.model as m
import tensorflow as tf
from tensorflow.keras import layers, models

# ==================> Classes
class cnn(m.Model):
    def __init__(self, x_train: list, y_train: list, x_test: list, y_test: list, dataset: str, seed: int) -> None:
        """__init__

        This method initializes the CNN class.

        Parameters:
            x_train: Training data
            y_train: Training labels
            x_test: Test data
            y_test: Test labels
            dataset: Dataset name
        Output:
            None
        """
        super().__init__(x_train, y_train, x_test, y_test, dataset, seed=seed)
        self.exe()

    # Override
    def expecific_model(self) -> object:
        """expecific_model

        This method is an override of the parent method for the CNN model.

        Output:
            object: CNN model
        """
        input_shape = (self.x_train.shape[1], 1)  # Thêm chiều cho CNN (nếu cần reshape)
        num_classes = len(set(self.y_train))  # Số lớp

        # Tạo model CNN
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
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )

        # Huấn luyện
        model.fit(self.x_train[..., None], self.y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.1)

        return model

    # Override
    def __str__(self) -> str:
        """__str__

        This method returns the name of the class.

        Output:
            str: Name of the class
        """
        return "cnn"
