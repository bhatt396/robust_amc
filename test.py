import torch
import numpy as np
from torch.utils.data import DataLoader

from config.config import Config
from data.dataset import IQDataset, load_rml_data
from data.generator import SignalGenerator
from models.moe_amc import MoEAMC
from utils.visualization import plot_iq_samples, plot_signal_spectrum

def test_single_signal(model, signal, config):
    """Test model on a single signal"""
    model.eval()
    
    # Normalize to unit power
    power = np.mean(np.abs(signal)**2)
    if power > 0:
        signal = signal / np.sqrt(power)
        
    # Prepare signal
    signal_tensor = np.stack([signal.real, signal.imag], axis=0)
    signal_tensor = torch.FloatTensor(signal_tensor).unsqueeze(0)
    signal_tensor = signal_tensor.to(config.DEVICE)
    
    with torch.no_grad():
        output = model(signal_tensor)
        prediction = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[prediction].item()
    
    return prediction, confidence, probs.cpu().numpy()

def test_model_interactive():
    """Interactive testing of the model"""
    config = Config()
    
    # Load model
    print("Loading MoE model...")
    model = MoEAMC(
        num_experts=config.NUM_EXPERTS,
        num_classes=config.NUM_CLASSES,
        input_channels=2,
        expert_filters=config.EXPERT_CNN_FILTERS
    ).to(config.DEVICE)
    
    try:
        model.load_state_dict(torch.load(f"{config.MODEL_PATH}/moe_amc_best.pth", map_location=config.DEVICE))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize signal generator
    generator = SignalGenerator(
        samples_per_symbol=config.SAMPLES_PER_SYMBOL,
        num_symbols=config.NUM_SYMBOLS
    )
    
    print("\n=== Interactive Testing Mode ===")
    print("Available modulations:", config.MODULATIONS)
    
    while True:
        print("\n" + "="*50)
        mod_input = input("Enter modulation type (e.g. QPSK, BPSK) or 'quit': ").strip().upper()
        
        if mod_input == 'QUIT':
            break
        
        if mod_input not in config.MODULATIONS:
            print(f"Invalid modulation! Available: {config.MODULATIONS}")
            continue
        
        snr_input = input(f"Enter SNR in dB ({config.SNR_RANGE[0]} to {config.SNR_RANGE[1]}): ").strip()
        try:
            snr = float(snr_input)
        except ValueError:
            print("Invalid SNR value!")
            continue
        
        # Generate signal
        signal = generator.generate_signal(mod_input)
        signal = generator.add_awgn(signal, snr)
        
        # Test
        prediction, confidence, probs = test_single_signal(model, signal, config)
        predicted_mod = config.MODULATIONS[prediction]
        
        # Results
        print(f"\nFinal Results:")
        print(f"True Modulation:      {mod_input}")
        print(f"Predicted Modulation: {predicted_mod}")
        print(f"Confidence:           {confidence*100:.2f}%")
        print(f"Result:               {'CORRECT' if predicted_mod == mod_input else 'WRONG'}")
        
        print("\nTop 3 Probabilities:")
        top_idx = np.argsort(probs)[-3:][::-1]
        for idx in top_idx:
            print(f"  {config.MODULATIONS[idx]}: {probs[idx]*100:.2f}%")

if __name__ == "__main__":
    test_model_interactive()