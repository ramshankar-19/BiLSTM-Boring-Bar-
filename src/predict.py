"""
Prediction and Testing script for BiLSTM vibration monitoring
(Without seaborn dependency)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model

def predict_vibration_state(model, X_new):
    """
    Make predictions on new data
    
    Returns:
    - predictions: class labels (0=stable, 1=transition, 2=violent)
    - probabilities: confidence scores
    """
    probabilities = model.predict(X_new, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    
    return predictions, probabilities

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix using matplotlib only"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix - BiLSTM Vibration Classification',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved to results/confusion_matrix.png")
    plt.close()

def main():
    print("="*70)
    print("BiLSTM Vibration State Prediction and Testing")
    print("="*70)
    
    # Load trained model
    model_path = '../models/final_bilstm_model.keras'
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Try alternate path
        model_path = '../models/best_bilstm_model.keras'
        if not os.path.exists(model_path):
            print(f"\nError: Model file not found!")
            print(f"Looking for: {model_path}")
            print("\nPlease train the model first by running: python3 train.py")
            return
    
    model = load_model(model_path)
    
    # Load test data
    print("\nLoading test data...")
    try:
        X_test = np.load('../data/processed/X_test.npy')
        y_test = np.load('../data/processed/y_test.npy')
        print(f"Test data loaded: {X_test.shape[0]} samples")
    except FileNotFoundError:
        print("\nError: Test data not found in data/processed/")
        print("Please run train.py first to generate test data.")
        return
    
    # Make predictions
    print("\nMaking predictions on test set...")
    predictions, probabilities = predict_vibration_state(model, X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    
    # Display results
    class_names = ['Stable', 'Transition', 'Violent Vibration']
    
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS (First 15 samples):")
    print(f"{'='*70}")
    print(f"{'Sample':<8} {'True Label':<20} {'Predicted':<20} {'Confidence':<12}")
    print("-"*70)
    
    for i in range(min(15, len(predictions))):
        true_label = class_names[y_test[i]]
        pred_label = class_names[predictions[i]]
        confidence = probabilities[i][predictions[i]] * 100
        
        # Add checkmark or X
        match = "✓" if predictions[i] == y_test[i] else "✗"
        print(f"{i+1:<8} {true_label:<20} {pred_label:<20} {confidence:>6.2f}%  {match}")
    
    # Overall accuracy
    print(f"\n{'='*70}")
    print(f"OVERALL TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("\n" + "="*70)
    print("CONFUSION MATRIX:")
    print("="*70)
    print(f"\n{'':>15} {'Predicted Stable':<18} {'Predicted Transition':<22} {'Predicted Violent':<18}")
    print(f"{'True Stable':<15} {cm[0,0]:<18} {cm[0,1]:<22} {cm[0,2]:<18}")
    print(f"{'True Transition':<15} {cm[1,0]:<18} {cm[1,1]:<22} {cm[1,2]:<18}")
    print(f"{'True Violent':<15} {cm[2,0]:<18} {cm[2,1]:<22} {cm[2,2]:<18}")
    
    # Classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT:")
    print("="*70)
    print(classification_report(y_test, predictions, target_names=class_names, digits=4))
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY:")
    print("="*70)
    for i, class_name in enumerate(class_names):
        class_correct = np.sum((predictions == i) & (y_test == i))
        class_total = np.sum(y_test == i)
        class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
        print(f"{class_name:<20}: {class_acc:>6.2f}% ({class_correct}/{class_total} correct)")
    
    # Plot confusion matrix
    try:
        plot_confusion_matrix(cm, class_names)
    except Exception as e:
        print(f"\nNote: Could not plot confusion matrix: {e}")
    
    # Save predictions
    print("\nSaving predictions...")
    os.makedirs('../results', exist_ok=True)
    np.save('../results/predictions.npy', predictions)
    np.save('../results/probabilities.npy', probabilities)
    print("Predictions saved to results/predictions.npy")
    
    print("\n" + "="*70)
    print("Testing completed successfully!")
    

if __name__ == "__main__":
    main()
