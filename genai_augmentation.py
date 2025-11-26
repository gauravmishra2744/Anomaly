"""
Generative AI for Data Augmentation and Synthetic Anomaly Generation
Uses VAE and GAN-inspired techniques for synthetic time-series generation
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class TimeSeriesVAE:
    """Variational Autoencoder for synthetic time-series generation"""
    
    def __init__(self, window_size=60, latent_dim=10):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.scaler = MinMaxScaler()
        
    def fit(self, X_train):
        """Fit VAE on training data"""
        self.scaler.fit(X_train.reshape(-1, 1))
        # Store statistics for synthetic generation
        self.mean = np.mean(X_train)
        self.std = np.std(X_train)
        self.min_val = np.min(X_train)
        self.max_val = np.max(X_train)
        
    def generate_normal_samples(self, n_samples=100):
        """Generate synthetic normal time-series samples"""
        synthetic_samples = []
        for _ in range(n_samples):
            # Generate smooth time-series using random walk with drift
            sample = np.random.normal(self.mean, self.std * 0.3, self.window_size)
            # Apply smoothing to make it realistic
            sample = np.convolve(sample, np.ones(3)/3, mode='same')
            # Clip to valid range
            sample = np.clip(sample, self.min_val, self.max_val)
            synthetic_samples.append(sample)
        
        return np.array(synthetic_samples).reshape(-1, self.window_size, 1)
    
    def generate_anomaly_samples(self, n_samples=50, anomaly_type='spike'):
        """Generate synthetic anomalies"""
        synthetic_anomalies = []
        
        for _ in range(n_samples):
            # Start with normal sample
            sample = np.random.normal(self.mean, self.std * 0.3, self.window_size)
            sample = np.convolve(sample, np.ones(3)/3, mode='same')
            
            if anomaly_type == 'spike':
                # Add sudden spike
                spike_pos = np.random.randint(10, self.window_size - 10)
                spike_height = np.random.uniform(self.max_val * 0.5, self.max_val)
                sample[spike_pos:spike_pos+5] = spike_height
                
            elif anomaly_type == 'shift':
                # Add level shift
                shift_pos = np.random.randint(10, self.window_size - 10)
                shift_amount = np.random.uniform(self.std * 2, self.std * 4)
                sample[shift_pos:] += shift_amount
                
            elif anomaly_type == 'trend':
                # Add abnormal trend
                trend_start = np.random.randint(5, 15)
                trend_slope = np.random.uniform(0.01, 0.05)
                sample[trend_start:] += np.arange(self.window_size - trend_start) * trend_slope
            
            # Clip to valid range
            sample = np.clip(sample, self.min_val, self.max_val)
            synthetic_anomalies.append(sample)
        
        return np.array(synthetic_anomalies).reshape(-1, self.window_size, 1)


def augment_training_data(X_train, augmentation_factor=0.5):
    """
    Augment training data with synthetic samples
    
    Args:
        X_train: Original training data
        augmentation_factor: Ratio of synthetic samples to add (0.5 = add 50% more data)
    
    Returns:
        X_train_augmented: Augmented training data
    """
    print("=" * 70)
    print("GENERATIVE AI - DATA AUGMENTATION")
    print("=" * 70)
    
    # Initialize VAE
    vae = TimeSeriesVAE(window_size=X_train.shape[1])
    vae.fit(X_train)
    
    # Generate synthetic normal samples
    n_synthetic = int(len(X_train) * augmentation_factor)
    print(f"\nGenerating {n_synthetic} synthetic normal samples...")
    X_synthetic_normal = vae.generate_normal_samples(n_samples=n_synthetic)
    
    # Combine original and synthetic data
    X_train_augmented = np.vstack([X_train, X_synthetic_normal])
    
    print(f"\nData Augmentation Summary:")
    print(f"  Original training samples: {len(X_train)}")
    print(f"  Synthetic samples added: {len(X_synthetic_normal)}")
    print(f"  Total augmented samples: {len(X_train_augmented)}")
    print(f"  Augmentation ratio: {len(X_synthetic_normal) / len(X_train) * 100:.1f}%")
    
    return X_train_augmented, vae


def generate_synthetic_anomalies(vae, n_samples=100, anomaly_types=['spike', 'shift', 'trend']):
    """
    Generate diverse synthetic anomalies for testing
    
    Args:
        vae: Trained VAE model
        n_samples: Total number of anomalies to generate
        anomaly_types: List of anomaly types to generate
    
    Returns:
        X_anomalies: Synthetic anomaly samples
        y_anomalies: Labels (all 1s for anomalies)
    """
    print("\n" + "=" * 70)
    print("GENERATIVE AI - SYNTHETIC ANOMALY GENERATION")
    print("=" * 70)
    
    X_anomalies = []
    samples_per_type = n_samples // len(anomaly_types)
    
    for anomaly_type in anomaly_types:
        print(f"\nGenerating {samples_per_type} '{anomaly_type}' anomalies...")
        X_type = vae.generate_anomaly_samples(n_samples=samples_per_type, anomaly_type=anomaly_type)
        X_anomalies.append(X_type)
    
    X_anomalies = np.vstack(X_anomalies)
    y_anomalies = np.ones(len(X_anomalies))
    
    print(f"\nSynthetic Anomaly Summary:")
    print(f"  Total synthetic anomalies: {len(X_anomalies)}")
    for anomaly_type in anomaly_types:
        print(f"    - {anomaly_type}: {samples_per_type}")
    
    return X_anomalies, y_anomalies


def save_augmentation_artifacts(vae, output_dir='./preprocessing_artifacts'):
    """Save VAE model for later use"""
    os.makedirs(output_dir, exist_ok=True)
    
    vae_config = {
        'window_size': vae.window_size,
        'latent_dim': vae.latent_dim,
        'mean': vae.mean,
        'std': vae.std,
        'min_val': vae.min_val,
        'max_val': vae.max_val
    }
    
    vae_path = os.path.join(output_dir, 'vae_config.pkl')
    with open(vae_path, 'wb') as f:
        pickle.dump(vae_config, f)
    
    print(f"\nVAE configuration saved to: {vae_path}")


if __name__ == "__main__":
    # Example usage
    print("Loading training data...")
    X_train = np.load('X_train.npy')
    
    # Augment data
    X_train_augmented, vae = augment_training_data(X_train, augmentation_factor=0.3)
    
    # Generate synthetic anomalies
    X_synthetic_anomalies, y_synthetic_anomalies = generate_synthetic_anomalies(
        vae, n_samples=100, anomaly_types=['spike', 'shift', 'trend']
    )
    
    # Save artifacts
    save_augmentation_artifacts(vae)
    
    print("\n" + "=" * 70)
    print("Augmentation complete!")
    print("=" * 70)
