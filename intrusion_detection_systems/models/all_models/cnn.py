# ========================== CNN Classifier =========================
#
#                   Author:  Sergio Arroni Del Riego
#
# =============================================================================
# ==================> Imports
import shared.models.model as m
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

# ==================> Classes
class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper for Keras CNN model to make it compatible with scikit-learn API.
    """
    def __init__(self, input_shape=None, num_classes=2, random_state=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def _build_model(self, input_shape):
        """
        Build the CNN model architecture
        """
        model = Sequential()
        
        # First Conv layer
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        
        # Second Conv layer
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        
        # Third Conv layer
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        
        # Flatten and Dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        
        # Output layer
        if self.num_classes > 2:
            model.add(Dense(self.num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        else:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y):
        """
        Fit the CNN model to the data
        """
        # Save original shape
        original_shape = X.shape
        # Step 1: Check if input is already 3D (if yes, flatten for scaler)
        if len(X.shape)==3:
            X_flat = X.reshape(X.shape[0],-1)
        else:
            X_flat = X # already 2D
        # Step 2: Scale features
        X_scaled = self.scaler.fit_transform(X_flat)
        # Step 3: Infer input shape if not set
        if self.input_shape is None:
            feature_len = X_scaled.shape[1]
            self.input_shape = (feature_len,1)
        # Step 4: Reshape for CNN input
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], self.input_shape[0], self.input_shape[1])
        
        # Handle classes
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        
        # Build model
        self.model = self._build_model((X_reshaped.shape[1], X_reshaped.shape[2]))
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.model.fit(
            X_reshaped, y,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """
        Predict classes for X
        """
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) ==3 else X
        X_scaled = self.scaler.transform(X_flat)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], self.input_shape[0], self.input_shape[1])
        
        # Get predictions
        if self.num_classes > 2:
            y_pred = np.argmax(self.model.predict(X_reshaped), axis=1)
        else:
            y_pred = (self.model.predict(X_reshaped) > 0.5).astype(int).flatten()
            
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X
        """
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
        X_scaled = self.scaler.transform(X_flat)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], self.input_shape[0], self.input_shape[1])
        
        # Get probabilities
        if self.num_classes > 2:
            y_proba = self.model.predict(X_reshaped)
        else:
            y_proba_single = self.model.predict(X_reshaped).flatten()
            y_proba = np.column_stack((1-y_proba_single, y_proba_single))
            
        return y_proba

class cnn(m.Model):
    def __init__(self, x_train: list, y_train: list, x_test: list, y_test: list, dataset: str, seed: int) -> None:
        """__init__
        This method is used to initialize the CNN class.
        Parameters:
            x_train: Training data
            y_train: Training labels
            x_test: Test data
            y_test: Test labels
            dataset: Dataset name
            seed: Random seed
        Output:
            None
        """
        super().__init__(x_train, y_train, x_test, y_test, dataset, seed=seed)
        self.exe()
    
    # Override
    def expecific_model(self) -> object:
        """expecific_model
        This method is an override of the parent method for the case of the CNN model.
        Output:
            object: CNN model
        """
        # Creating and returning the CNN classifier
        cnn_model = CNNClassifier(random_state=self.seed)
        
        # Reshape data for CNN if needed (detecting whether reshaping has already been done)
        x_train = self.x_train
        if len(x_train.shape) == 2:
            # Feature data needs to be reshaped for CNN
            pass  # Reshaping will be handled internally in the classifier
            
        cnn_model.fit(self.x_train, self.y_train)
        return cnn_model
    
    # Override
    def __str__(self) -> str:
        """__str__
        This method is used to return the name of the class.
        Output:
            str: Name of the class
        """
        return "cnn"