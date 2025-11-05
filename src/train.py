"""
Training script for BiLSTM vibration monitoring
WITH PROPER EXPERIMENT-LEVEL SPLITTING (No Data Leakage)
"""
import numpy as np
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src directory to path
sys.path.append(os.path.dirname(__file__))

from data_generator import BiLSTMDatasetGenerator
from bilstm_model import BoringBarBiLSTM
from generate_realistic_data import generate_realistic_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def main():
    print("="*70)
    print("BiLSTM Boring Bar Vibration Monitoring - Training")
    print("(With Proper Experiment-Level Splitting)")
    print("="*70)
    
    # Step 1: Generate realistic dataset
    print("\n[1/6] Generating realistic dataset...")
    data_dict = generate_realistic_dataset(num_experiments=192, signal_length=40000)
    
    num_experiments = len(data_dict['labels'])
    print(f"Total experiments generated: {num_experiments}")
    print(f"Class distribution: {np.bincount(data_dict['labels'])}")
    
    # Step 2: Split at EXPERIMENT level (before windowing)
    print("\n[2/6] Splitting experiments into train/test sets...")
    print("=" * 70)
    print("IMPORTANT: Splitting EXPERIMENTS before windowing to prevent leakage")
    print("=" * 70)
    
    experiment_ids = np.arange(num_experiments)
    
    # Split experiments 70/30
    train_exp_ids, test_exp_ids = train_test_split(
        experiment_ids,
        test_size=0.3,
        random_state=42,
        stratify=data_dict['labels']
    )
    
    print(f"Train experiments: {len(train_exp_ids)} (70%)")
    print(f"Test experiments: {len(test_exp_ids)} (30%)")
    
    # Create separate train and test dictionaries
    train_dict = {
        'accel_x': [data_dict['accel_x'][i] for i in train_exp_ids],
        'accel_y': [data_dict['accel_y'][i] for i in train_exp_ids],
        'accel_z': [data_dict['accel_z'][i] for i in train_exp_ids],
        'sound_pressure': [data_dict['sound_pressure'][i] for i in train_exp_ids],
        'labels': data_dict['labels'][train_exp_ids]
    }
    
    test_dict = {
        'accel_x': [data_dict['accel_x'][i] for i in test_exp_ids],
        'accel_y': [data_dict['accel_y'][i] for i in test_exp_ids],
        'accel_z': [data_dict['accel_z'][i] for i in test_exp_ids],
        'sound_pressure': [data_dict['sound_pressure'][i] for i in test_exp_ids],
        'labels': data_dict['labels'][test_exp_ids]
    }
    
    print(f"Train experiments class distribution: {np.bincount(train_dict['labels'])}")
    print(f"Test experiments class distribution: {np.bincount(test_dict['labels'])}")
    
    # Step 3: Create windows separately for train and test
    print("\n[3/6] Creating windowed features from experiments...")
    
    generator = BiLSTMDatasetGenerator(
        sampling_freq=40000,
        window_size=1000,
        overlap=0.5
    )
    
    # Generate train windows
    print("Processing training experiments...")
    X_train, y_train = generator.generate_dataset(train_dict)
    
    # Generate test windows
    print("Processing test experiments...")
    X_test, y_test = generator.generate_dataset(test_dict)
    
    print(f"\nWindowed datasets created:")
    print(f"Training windows: X={X_train.shape}, y={y_train.shape}")
    print(f"Test windows: X={X_test.shape}, y={y_test.shape}")
    
    # Step 4: Split training into train/validation
    print("\n[4/6] Creating validation set from training data...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    print(f"Final training set: {X_train_split.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Data quality check
    print("\n" + "="*70)
    print("DATA QUALITY & LEAKAGE CHECK:")
    print("="*70)
    print(f"Training samples: {len(y_train_split)} | Classes: {np.bincount(y_train_split)}")
    print(f"Validation samples: {len(y_val)} | Classes: {np.bincount(y_val)}")
    print(f"Test samples: {len(y_test)} | Classes: {np.bincount(y_test)}")
    
    print("\n✓ Experiments split BEFORE windowing - No leakage possible!")
    print("✓ Test set contains windows from completely different experiments")
    print("="*70)
    
    # Save datasets
    print("\nSaving processed datasets...")
    os.makedirs('../data/processed', exist_ok=True)
    np.save('../data/processed/X_train.npy', X_train_split)
    np.save('../data/processed/X_val.npy', X_val)
    np.save('../data/processed/X_test.npy', X_test)
    np.save('../data/processed/y_train.npy', y_train_split)
    np.save('../data/processed/y_val.npy', y_val)
    np.save('../data/processed/y_test.npy', y_test)
    print("Datasets saved to data/processed/")
    
    # Step 5: Build model
    print("\n[5/6] Building BiLSTM model...")
    timesteps = X_train_split.shape[1]
    num_features = X_train_split.shape[2]
    
    bilstm_model = BoringBarBiLSTM(
        timesteps=timesteps,
        num_features=num_features,
        num_classes=3
    )
    
    model = bilstm_model.build_model(lstm_units=[128, 64], dropout_rate=0.3)
    bilstm_model.compile_model(learning_rate=0.1, momentum=0.9)
    
    print("\nModel Architecture:")
    bilstm_model.summary()
    
    # Step 6: Train model
    print("\n[6/6] Training model...")
    print("="*70)
    print("Starting training... This may take a while.")
    print("="*70)
    
    history = bilstm_model.train(
        X_train_split, y_train_split,
        X_val, y_val,
        batch_size=64,
        epochs=20,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_loss, test_accuracy = bilstm_model.evaluate(X_test, y_test)
    
    print(f"\n{'='*70}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    # Generate detailed predictions and metrics
    print("\nGenerating detailed classification metrics...")
    predictions = bilstm_model.model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['Stable', 'Transition', 'Violent']
    
    # Print confusion matrix
    print("\n" + "="*70)
    print("CONFUSION MATRIX:")
    print("="*70)
    print(f"\n{'':>15} {'Pred Stable':<15} {'Pred Transition':<18} {'Pred Violent':<15}")
    print(f"{'True Stable':<15} {cm[0,0]:<15} {cm[0,1]:<18} {cm[0,2]:<15}")
    print(f"{'True Transition':<15} {cm[1,0]:<15} {cm[1,1]:<18} {cm[1,2]:<15}")
    print(f"{'True Violent':<15} {cm[2,0]:<15} {cm[2,1]:<18} {cm[2,2]:<15}")
    
    # Classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT:")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY:")
    print("="*70)
    for i, class_name in enumerate(class_names):
        class_correct = np.sum((y_pred == i) & (y_test == i))
        class_total = np.sum(y_test == i)
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        print(f"{class_name:<20}: {class_acc:>6.2f}% ({class_correct}/{class_total} correct)")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix visualization...")
    os.makedirs('../results', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Count')
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, 
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    ax.set_title('Confusion Matrix - BiLSTM Vibration Classification\n(No Data Leakage)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig('../results/confusion_matrix_training.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to results/confusion_matrix_training.png")
    plt.close()
    
    # Plot training history
    print("\nGenerating training history plots...")
    bilstm_model.plot_training_history()
    
    # Save final model
    print("\nSaving trained model...")
    os.makedirs('../models', exist_ok=True)
    bilstm_model.model.save('../models/final_bilstm_model.keras')
    print("Model saved to models/final_bilstm_model.keras")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"✓ Model trained with proper experiment-level splitting")
    print(f"✓ Test accuracy: {test_accuracy*100:.2f}%")
    print(f"✓ Model saved successfully")
    print(f"✓ Confusion matrix and training plots saved")
    print("\nExpected accuracy range: 70-85% (realistic for generalization)")
    print("100% accuracy would indicate overfitting or data leakage")
    print("="*70)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
