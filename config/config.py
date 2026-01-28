import os
import torch

class Config:
    # Dataset source: 'generated' or 'rml'
    DATA_SOURCE = 'rml' 
    RML_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'RML2016.10a_dict.pkl')
    
    # Dataset parameters
    GENERATED_MODULATIONS = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'GFSK', 'CPFSK', 'PAM4']
    RML_MODULATIONS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
    MODULATIONS = RML_MODULATIONS if DATA_SOURCE == 'rml' else GENERATED_MODULATIONS
    NUM_CLASSES = len(MODULATIONS)
    SAMPLES_PER_SYMBOL = 8
    NUM_SYMBOLS = 128
    SAMPLE_LENGTH = 128 if DATA_SOURCE == 'rml' else (SAMPLES_PER_SYMBOL * NUM_SYMBOLS)
    
    # SNR parameters
    SNR_RANGE = (-20, 18)  # RadioML 2016.10a range is usually -20 to 18 dB
    SNR_LOW = (-20, 0)
    SNR_MID = (0, 10)
    SNR_HIGH = (10, 20)
    SNR_BINS = ['low', 'mid', 'high']
    NUM_EXPERTS = len(SNR_BINS)
    
    # Channel models
    CHANNELS = ['AWGN', 'Rayleigh', 'Rician']
    
    # Training parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    
    # Model parameters
    SNR_ESTIMATOR_HIDDEN = [128, 64]
    EXPERT_CNN_FILTERS = [64, 128, 256]
    GATING_HIDDEN = [128, 64]
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data')
    MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints')
    LOG_PATH = os.path.join(BASE_DIR, 'logs')
    RESULTS_PATH = os.path.join(BASE_DIR, 'results')
    
    @classmethod
    def get_snr_bin(cls, snr):
        """Convert SNR value to bin index"""
        if snr < cls.SNR_LOW[1]:
            return 0  # low
        elif snr < cls.SNR_MID[1]:
            return 1  # mid
        else:
            return 2  # high