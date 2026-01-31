import numpy as np
from scipy import signal as sp_signal

class SignalGenerator:
    def __init__(self, samples_per_symbol=8, num_symbols=128):
        self.samples_per_symbol = samples_per_symbol
        self.num_symbols = num_symbols
        self.sample_length = samples_per_symbol * num_symbols
        
    def generate_bpsk(self):
        """Generate BPSK signal"""
        bits = np.random.randint(0, 2, self.num_symbols)
        symbols = 2 * bits - 1
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal.astype(np.complex64)
    
    def generate_qpsk(self):
        """Generate QPSK signal"""
        bits = np.random.randint(0, 4, self.num_symbols)
        constellation = np.exp(1j * (2 * np.pi * bits / 4 + np.pi / 4))
        signal = np.repeat(constellation, self.samples_per_symbol)
        return signal
    
    def generate_8psk(self):
        """Generate 8PSK signal"""
        bits = np.random.randint(0, 8, self.num_symbols)
        constellation = np.exp(1j * (2 * np.pi * bits / 8 + np.pi / 8))
        signal = np.repeat(constellation, self.samples_per_symbol)
        return signal
    
    def generate_qam16(self):
        """Generate 16-QAM signal"""
        bits = np.random.randint(0, 16, self.num_symbols)
        I = 2 * (bits % 4) - 3
        Q = 2 * (bits // 4) - 3
        constellation = (I + 1j * Q) / np.sqrt(10)
        signal = np.repeat(constellation, self.samples_per_symbol)
        return signal
    
    def generate_qam64(self):
        """Generate 64-QAM signal"""
        bits = np.random.randint(0, 64, self.num_symbols)
        I = 2 * (bits % 8) - 7
        Q = 2 * (bits // 8) - 7
        constellation = (I + 1j * Q) / np.sqrt(42)
        signal = np.repeat(constellation, self.samples_per_symbol)
        return signal
    
    def generate_gfsk(self, bt=0.5):
        """Generate GFSK signal"""
        bits = np.random.randint(0, 2, self.num_symbols)
        bits = 2 * bits - 1
        
        # Gaussian filter
        t = np.arange(-4, 4, 1/self.samples_per_symbol)
        h = np.exp(-2 * np.pi**2 * bt**2 * t**2)
        h = h / np.sum(h)
        
        upsampled = np.zeros(len(bits) * self.samples_per_symbol)
        upsampled[::self.samples_per_symbol] = bits
        filtered = np.convolve(upsampled, h, mode='same')
        
        phase = np.cumsum(filtered) * np.pi / 2
        signal = np.exp(1j * phase)
        return signal[:self.sample_length]
    
    def generate_cpfsk(self, mod_index=0.5):
        """Generate CPFSK signal"""
        bits = np.random.randint(0, 2, self.num_symbols)
        bits = 2 * bits - 1
        
        upsampled = np.repeat(bits, self.samples_per_symbol)
        phase = np.cumsum(upsampled) * np.pi * mod_index
        signal = np.exp(1j * phase)
        return signal
    
    def generate_pam4(self):
        """Generate PAM4 signal"""
        bits = np.random.randint(0, 4, self.num_symbols)
        symbols = 2 * bits - 3
        signal = np.repeat(symbols, self.samples_per_symbol)
        return signal.astype(np.complex64)
    
    def add_awgn(self, signal, snr_db):
        """Add AWGN to signal"""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
        return signal + noise
    
    def add_rayleigh_fading(self, signal):
        """Add Rayleigh fading"""
        h = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) / np.sqrt(2)
        return signal * h
    
    def add_rician_fading(self, signal, k_factor=10):
        """Add Rician fading"""
        k_linear = 10**(k_factor / 10)
        los = np.sqrt(k_linear / (k_linear + 1))
        nlos = np.sqrt(1 / (k_linear + 1)) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        ) / np.sqrt(2)
        h = los + nlos
        return signal * h
    
    def generate_signal(self, modulation):
        """Generate signal based on modulation type"""
        mod_functions = {
            'BPSK': self.generate_bpsk,
            'QPSK': self.generate_qpsk,
            '8PSK': self.generate_8psk,
            'QAM16': self.generate_qam16,
            'QAM64': self.generate_qam64,
            'GFSK': self.generate_gfsk,
            'CPFSK': self.generate_cpfsk,
            'PAM4': self.generate_pam4
        }
        return mod_functions[modulation]()
    
    def generate_dataset(self, modulations, snr_range, samples_per_mod=1000, channel='AWGN'):
        """Generate complete dataset"""
        signals = []
        labels = []
        snr_values = []
        
        for mod_idx, modulation in enumerate(modulations):
            for _ in range(samples_per_mod):
                sig = self.generate_signal(modulation)
                snr = np.random.uniform(snr_range[0], snr_range[1])
                
                if channel == 'AWGN':
                    sig = self.add_awgn(sig, snr)
                elif channel == 'Rayleigh':
                    sig = self.add_rayleigh_fading(sig)
                    sig = self.add_awgn(sig, snr)
                elif channel == 'Rician':
                    sig = self.add_rician_fading(sig)
                    sig = self.add_awgn(sig, snr)
                
                # Normalize
                sig = sig / np.max(np.abs(sig))
                
                signals.append(sig)
                labels.append(mod_idx)
                snr_values.append(snr)
        
        return np.array(signals), np.array(labels), np.array(snr_values)