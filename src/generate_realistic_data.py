"""
Generate synthetic vibration data with realistic patterns
for testing the BiLSTM model
"""
import numpy as np
import os

def generate_stable_signal(length, sampling_freq=40000):
    """Generate stable cutting signal - low amplitude, low frequency"""
    t = np.linspace(0, length/sampling_freq, length)
    
    # Low frequency vibration (50-100 Hz)
    signal = 0.1 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.sin(2 * np.pi * 80 * t)
    
    # Small random noise
    signal += 0.02 * np.random.randn(length)
    
    return signal

def generate_transition_signal(length, sampling_freq=40000):
    """Generate transition state - medium amplitude, increasing frequency"""
    t = np.linspace(0, length/sampling_freq, length)
    
    # Medium frequency vibration (100-300 Hz) with amplitude modulation
    base_freq = 150
    signal = 0.3 * np.sin(2 * np.pi * base_freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    
    # Add harmonics
    signal += 0.15 * np.sin(2 * np.pi * 250 * t)
    
    # Medium noise
    signal += 0.05 * np.random.randn(length)
    
    return signal

def generate_violent_signal(length, sampling_freq=40000):
    """Generate violent vibration - high amplitude, high frequency"""
    t = np.linspace(0, length/sampling_freq, length)
    
    # High frequency vibration (300-800 Hz) with chaos
    signal = 0.8 * np.sin(2 * np.pi * 500 * t)
    signal += 0.5 * np.sin(2 * np.pi * 700 * t)
    
    # Add chaotic component
    signal += 0.3 * np.sin(2 * np.pi * 350 * t) * np.sin(2 * np.pi * 15 * t)
    
    # High noise (chatter)
    signal += 0.15 * np.random.randn(length)
    
    # Random spikes
    spike_indices = np.random.choice(length, size=int(length*0.01), replace=False)
    signal[spike_indices] += np.random.uniform(0.5, 1.0, len(spike_indices))
    
    return signal

def generate_realistic_dataset(num_experiments=192, signal_length=40000):
    """
    Generate balanced dataset with realistic vibration patterns
    """
    print("Generating realistic synthetic vibration dataset...")
    
    # Ensure balanced classes
    samples_per_class = num_experiments // 3
    
    data_dict = {
        'accel_x': [],
        'accel_y': [],
        'accel_z': [],
        'sound_pressure': [],
        'labels': []
    }
    
    # Generate for each class
    for class_label in range(3):
        print(f"Generating class {class_label} ({['Stable', 'Transition', 'Violent'][class_label]})...")
        
        for i in range(samples_per_class):
            if class_label == 0:  # Stable
                accel_x = generate_stable_signal(signal_length)
                accel_y = generate_stable_signal(signal_length) * 0.8
                accel_z = generate_stable_signal(signal_length) * 0.6
                sound = generate_stable_signal(signal_length) * 1.2
                
            elif class_label == 1:  # Transition
                accel_x = generate_transition_signal(signal_length)
                accel_y = generate_transition_signal(signal_length) * 0.9
                accel_z = generate_transition_signal(signal_length) * 0.7
                sound = generate_transition_signal(signal_length) * 1.5
                
            else:  # Violent
                accel_x = generate_violent_signal(signal_length)
                accel_y = generate_violent_signal(signal_length) * 1.1
                accel_z = generate_violent_signal(signal_length) * 0.8
                sound = generate_violent_signal(signal_length) * 2.0
            
            data_dict['accel_x'].append(accel_x)
            data_dict['accel_y'].append(accel_y)
            data_dict['accel_z'].append(accel_z)
            data_dict['sound_pressure'].append(sound)
            data_dict['labels'].append(class_label)
    
    # Shuffle the data
    indices = np.arange(len(data_dict['labels']))
    np.random.shuffle(indices)
    
    for key in data_dict.keys():
        if key != 'labels':
            data_dict[key] = [data_dict[key][i] for i in indices]
        else:
            data_dict[key] = np.array([data_dict[key][i] for i in indices])
    
    print(f"\nDataset generated: {len(data_dict['labels'])} samples")
    print(f"Class distribution: {np.bincount(data_dict['labels'])}")
    experiment_ids = []
    for i in range(num_experiments):
        experiment_ids.append(i)
    
    data_dict['experiment_ids'] = np.array(experiment_ids)
    
    return data_dict

if __name__ == "__main__":
    # Generate realistic data
    data_dict = generate_realistic_dataset(num_experiments=192, signal_length=40000)
    
    # Save to file
    os.makedirs('../data/raw', exist_ok=True)
    np.save('../data/raw/realistic_data.npy', data_dict)
    print("\nRealistic dataset saved to data/raw/realistic_data.npy")
