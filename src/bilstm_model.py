import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

class BoringBarBiLSTM:
    """
    BiLSTM network for boring bar vibration state classification.
    Based on the Shuffle-BiLSTM paper architecture (BiLSTM component only).
    """
    
    def __init__(self, timesteps, num_features, num_classes=3):
        """
        Parameters:
        -----------
        timesteps : int
            Number of time steps in sequence
        num_features : int
            Number of features per timestep
        num_classes : int
            Number of output classes (3: stable, transition, violent)
        """
        self.timesteps = timesteps
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[128, 64], dropout_rate=0.3):
        """
        Build BiLSTM network architecture.
        
        Parameters:
        -----------
        lstm_units : list
            Number of units in each BiLSTM layer
        dropout_rate : float
            Dropout rate for regularization
        """
        # Input layer
        inputs = layers.Input(shape=(self.timesteps, self.num_features), name='input_layer')
        
        # First BiLSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units[0],
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                name='lstm_1'
            ),
            name='bidirectional_lstm_1'
        )(inputs)
        
        # Batch normalization
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        
        # Dropout for regularization
        x = layers.Dropout(dropout_rate, name='dropout_1')(x)
        
        # Second BiLSTM layer (no return_sequences since we want final output)
        x = layers.Bidirectional(
            layers.LSTM(
                lstm_units[1],
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                name='lstm_2'
            ),
            name='bidirectional_lstm_2'
        )(x)
        
        # Batch normalization
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        
        # Dropout
        x = layers.Dropout(dropout_rate, name='dropout_2')(x)
        
        # Fully connected layer
        x = layers.Dense(128, activation='relu', name='fc_layer')(x)
        x = layers.BatchNormalization(name='batch_norm_3')(x)
        x = layers.Dropout(dropout_rate/2, name='dropout_3')(x)
        
        # Output layer with SoftMax classifier (as per paper)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output_layer')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='BiLSTM_VibrationMonitor')
        
        return self.model
    
    def compile_model(self, learning_rate=0.1, momentum=0.9):
        """
        Compile model with SGDM optimizer (as per paper).
        
        Parameters:
        -----------
        learning_rate : float
            Initial learning rate (0.1 as per paper)
        momentum : float
            Momentum for SGD (0.9 as per paper)
        """
        # SGDM optimizer as specified in paper
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully!")
        return self.model
    
    def get_callbacks(self):
        """
        Set up training callbacks for learning rate scheduling and early stopping.
        Based on paper parameters: LR reduced by 0.1 every 100 iterations.
        """
        # Learning rate scheduler (multiply by 0.1 every 100 epochs as per paper)
        def lr_schedule(epoch, lr):
            if epoch > 0 and epoch % 100 == 0:
                return lr * 0.1
            return lr
        
        lr_scheduler = callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        
        # Early stopping to prevent overfitting
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint to save best model - FIXED: use .keras extension
        checkpoint = callbacks.ModelCheckpoint(
            'best_bilstm_model.keras',  # Changed from .h5 to .keras
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=20,
            min_lr=1e-7,
            verbose=1
        )
        
        return [lr_scheduler, early_stop, checkpoint, reduce_lr]
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=64, epochs=1000, verbose=1):
        """
        Train the BiLSTM model.
        
        Parameters:
        -----------
        X_train : array
            Training data of shape (samples, timesteps, features)
        y_train : array
            Training labels
        X_val : array
            Validation data
        y_val : array
            Validation labels
        batch_size : int
            Batch size (64 as per paper)
        epochs : int
            Maximum epochs (1000 as per paper)
        verbose : int
            Verbosity level
        """
        # Get callbacks
        callback_list = self.get_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        
        Returns:
        --------
        test_loss : float
        test_accuracy : float
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        return test_loss, test_accuracy
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Returns:
        --------
        predictions : array
            Predicted class labels
        probabilities : array
            Class probabilities for each sample
        """
        probabilities = self.model.predict(X)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def get_confusion_matrix(self, X_test, y_test):
        """
        Generate confusion matrix for test set.
        """
        from sklearn.metrics import confusion_matrix, classification_report
        
        predictions, _ = self.predict(X_test)
        
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        class_names = ['Stable (0)', 'Transition (1)', 'Violent (2)']
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=class_names))
        
        return cm
    
    def plot_training_history(self):
        """
        Plot training and validation accuracy/loss curves.
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def summary(self):
        """
        Print model summary.
        """
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        self.model.summary()
