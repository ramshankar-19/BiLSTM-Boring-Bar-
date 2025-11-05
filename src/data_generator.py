import numpy as np
import pandas as pd
import pywt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BiLSTMDatasetGenerator:
    """
    Generate sequential time-series dataset for BiLSTM from vibration sensor data.
    Based on the boring bar vibration monitoring paper.
    """
    
    def __init__(self, sampling_freq=40000, window_size=1000, overlap=0.5):
        """
        Parameters:
        -----------
        sampling_freq : int
            Sampling frequency (40 kHz as per paper)
        window_size : int
            Number of samples per window
        overlap : float
            Overlap ratio between consecutive windows (0 to 1)
        """
        self.sampling_freq = sampling_freq
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
    def wavelet_denoise(self, signal_data, wavelet='coif5', level=3):
        """
        Apply wavelet packet threshold denoising (as per paper).
        
        Parameters:
        -----------
        signal_data : array
            Raw signal data
        wavelet : str
            Wavelet basis function (coif5 as per paper)
        level : int
            Decomposition level (3 layers as per paper)
        """
        # Decompose signal using wavelet packet
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Calculate threshold using unbiased likelihood estimation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # Apply hard thresholding
        denoised_coeffs = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
        
        # Reconstruct signal
        denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
        
        return denoised_signal[:len(signal_data)]
    
    def extract_time_domain_features(self, window):
        """
        Extract statistical time-domain features from signal window.
        
        Returns 8 features per window.
        """
        features = []
        features.append(np.mean(window))                    # Mean
        features.append(np.std(window))                     # Standard deviation
        features.append(np.sqrt(np.mean(window**2)))        # RMS
        features.append(np.max(window) - np.min(window))    # Peak-to-peak
        features.append(np.max(np.abs(window)))             # Peak amplitude
        
        # Kurtosis (4th moment)
        if np.std(window) != 0:
            features.append(np.mean((window - np.mean(window))**4) / np.std(window)**4)
        else:
            features.append(0)
        
        # Skewness (3rd moment)
        if np.std(window) != 0:
            features.append(np.mean((window - np.mean(window))**3) / np.std(window)**3)
        else:
            features.append(0)
        
        # Crest factor
        rms = np.sqrt(np.mean(window**2))
        features.append(np.max(np.abs(window)) / rms if rms != 0 else 0)
        
        return features
    
    def extract_frequency_domain_features(self, window):
        """
        Extract frequency-domain features using FFT.
        
        Returns 6 features per window.
        """
        # Compute FFT
        fft_vals = np.fft.rfft(window)
        fft_mag = np.abs(fft_vals)
        fft_freq = np.fft.rfftfreq(len(window), 1/self.sampling_freq)
        
        features = []
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_mag)
        features.append(fft_freq[dominant_freq_idx])
        
        # Spectral energy
        features.append(np.sum(fft_mag**2))
        
        # Spectral centroid
        features.append(np.sum(fft_freq * fft_mag) / np.sum(fft_mag) if np.sum(fft_mag) != 0 else 0)
        
        # Energy in different frequency bands (0-1kHz, 1-5kHz, 5-10kHz)
        band1 = np.sum(fft_mag[(fft_freq >= 0) & (fft_freq < 1000)]**2)
        band2 = np.sum(fft_mag[(fft_freq >= 1000) & (fft_freq < 5000)]**2)
        band3 = np.sum(fft_mag[(fft_freq >= 5000) & (fft_freq <= 10000)]**2)
        
        features.extend([band1, band2, band3])
        
        return features
    
    def create_sequences(self, accel_x, accel_y, accel_z, sound_pressure, label):
        """
        Create sequential feature vectors for BiLSTM from multi-sensor data.
        
        Parameters:
        -----------
        accel_x, accel_y, accel_z : array
            Three-axis acceleration signals
        sound_pressure : array
            Sound pressure signal
        label : int
            Class label (0: stable, 1: transition, 2: violent vibration)
        
        Returns:
        --------
        sequences : array of shape (num_sequences, timesteps, features)
        labels : array of shape (num_sequences,)
        """
        # Apply wavelet denoising to all signals
        accel_x_clean = self.wavelet_denoise(accel_x)
        accel_y_clean = self.wavelet_denoise(accel_y)
        accel_z_clean = self.wavelet_denoise(accel_z)
        sound_clean = self.wavelet_denoise(sound_pressure)
        
        sequences = []
        
        # Create sliding windows
        num_windows = (len(accel_x_clean) - self.window_size) // self.step_size + 1
        
        for i in range(num_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            
            # Extract windows for all sensors
            window_x = accel_x_clean[start_idx:end_idx]
            window_y = accel_y_clean[start_idx:end_idx]
            window_z = accel_z_clean[start_idx:end_idx]
            window_sound = sound_clean[start_idx:end_idx]
            
            # Extract features for each sensor
            features_x_time = self.extract_time_domain_features(window_x)
            features_x_freq = self.extract_frequency_domain_features(window_x)
            
            features_y_time = self.extract_time_domain_features(window_y)
            features_y_freq = self.extract_frequency_domain_features(window_y)
            
            features_z_time = self.extract_time_domain_features(window_z)
            features_z_freq = self.extract_frequency_domain_features(window_z)
            
            features_sound_time = self.extract_time_domain_features(window_sound)
            features_sound_freq = self.extract_frequency_domain_features(window_sound)
            
            # Concatenate all features (4 sensors × 14 features = 56 features per timestep)
            timestep_features = (features_x_time + features_x_freq +
                               features_y_time + features_y_freq +
                               features_z_time + features_z_freq +
                               features_sound_time + features_sound_freq)
            
            sequences.append(timestep_features)
        
        # Reshape to (1, num_timesteps, num_features)
        sequences = np.array(sequences).reshape(1, -1, 56)
        labels = np.array([label])
        
        return sequences, labels
    
    def generate_dataset(self, data_dict):
        """
        Generate complete dataset from multiple experiments.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary with keys:
            - 'accel_x': list of arrays (one per experiment)
            - 'accel_y': list of arrays
            - 'accel_z': list of arrays
            - 'sound_pressure': list of arrays
            - 'labels': list of integers (0, 1, or 2)
        
        Returns:
        --------
        X : array of shape (num_samples, timesteps, features)
        y : array of shape (num_samples,)
        """
        all_sequences = []
        all_labels = []
        
        num_experiments = len(data_dict['labels'])
        
        for i in range(num_experiments):
            sequences, labels = self.create_sequences(
                data_dict['accel_x'][i],
                data_dict['accel_y'][i],
                data_dict['accel_z'][i],
                data_dict['sound_pressure'][i],
                data_dict['labels'][i]
            )
            all_sequences.append(sequences)
            all_labels.append(labels)
        
        X = np.vstack(all_sequences)
        y = np.concatenate(all_labels)
        
        return X, y


# Example usage:
if __name__ == "__main__":
    # Simulate data (replace with actual sensor data)
    # Each experiment collected at 40 kHz
    num_experiments = 192
    signal_length = 40000  # 1 second of data
    
    data_dict = {
        'accel_x': [np.random.randn(signal_length) for _ in range(num_experiments)],
        'accel_y': [np.random.randn(signal_length) for _ in range(num_experiments)],
        'accel_z': [np.random.randn(signal_length) for _ in range(num_experiments)],
        'sound_pressure': [np.random.randn(signal_length) for _ in range(num_experiments)],
        'labels': np.random.randint(0, 3, num_experiments)  # 0: stable, 1: transition, 2: violent
    }
    
    # Generate dataset
    generator = BiLSTMDatasetGenerator(sampling_freq=40000, window_size=1000, overlap=0.5)
    X, y = generator.generate_dataset(data_dict)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features per timestep: {X.shape[2]}")
    print(f"Number of timesteps: {X.shape[1]}")
    
    # Split into train/test (70/30 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save dataset
    np.save('X_train_bilstm.npy', X_train)
    np.save('X_test_bilstm.npy', X_test)
    np.save('y_train_bilstm.npy', y_train)
    np.save('y_test_bilstm.npy', y_test)
    
    print("Dataset saved successfully!")
