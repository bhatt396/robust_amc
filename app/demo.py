"""Web demo using Gradio or Streamlit"""
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from models.moe_model import SNRAwareMoE
from utils.config import config

def load_model():
    """Load trained MoE model"""
    model = SNRAwareMoE(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load("saved_models/best_moe_model.pth", 
                                    map_location='cpu'))
    model.eval()
    return model

def generate_signal(modulation, snr):
    """Generate a sample signal with given modulation and SNR"""
    # This is a simplified version - in real demo, you'd use actual signal generation
    # or load from dataset
    t = np.linspace(0, 1, 128)
    
    if modulation == "BPSK":
        i = np.cos(2 * np.pi * t) * (np.random.choice([-1, 1], size=128))
        q = np.zeros(128)
    elif modulation == "QPSK":
        i = np.cos(2 * np.pi * t) * (np.random.choice([-1, 1], size=128))
        q = np.sin(2 * np.pi * t) * (np.random.choice([-1, 1], size=128))
    else:
        i = np.cos(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t)
        q = np.sin(2 * np.pi * t)
    
    # Add noise based on SNR
    noise_power = 10 ** (-snr / 20)
    i_noisy = i + noise_power * np.random.randn(128)
    q_noisy = q + noise_power * np.random.randn(128)
    
    signal = np.stack([i_noisy, q_noisy], axis=0)
    return signal

def classify_signal(signal, model):
    """Classify the signal using MoE model"""
    with torch.no_grad():
        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        predictions, probabilities, expert_used = model.predict(signal_tensor)
    
    # Modulation labels (RML2016.10a order)
    mod_labels = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 
                  'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
    pred_idx = predictions.item()
    confidence = torch.max(probabilities).item()
    expert_name = ['Low SNR', 'Mid SNR', 'High SNR'][expert_used.item()]
    
    return mod_labels[pred_idx], f"{confidence*100:.1f}%", expert_name

def create_demo():
    """Create Gradio interface"""
    model = load_model()
    
    def process_signal(modulation, snr):
        # Generate signal
        signal = generate_signal(modulation, snr)
        
        # Classify
        pred_mod, confidence, expert = classify_signal(signal, model)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot I and Q components
        axes[0, 0].plot(signal[0], 'b-', label='I component')
        axes[0, 0].set_title('In-Phase Component (I)')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(signal[1], 'r-', label='Q component')
        axes[0, 1].set_title('Quadrature Component (Q)')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot constellation
        axes[1, 0].scatter(signal[0], signal[1], alpha=0.6)
        axes[1, 0].set_title('Constellation Diagram')
        axes[1, 0].set_xlabel('I')
        axes[1, 0].set_ylabel('Q')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Add text info
        axes[1, 1].axis('off')
        info_text = f"""
        Input Parameters:
        - Modulation: {modulation}
        - SNR: {snr} dB
        
        Model Prediction:
        - Predicted: {pred_mod}
        - Confidence: {confidence}
        - Expert Used: {expert}
        
        {'✓ Correct!' if pred_mod == modulation else '✗ Incorrect'}
        """
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, 
                       verticalalignment='center')
        
        plt.tight_layout()
        return fig, pred_mod, confidence, expert
    
    # Create interface
    with gr.Blocks(title="SNR-Adaptive AMC Demo") as demo:
        gr.Markdown("# SNR-Adaptive Modulation Classification Demo")
        gr.Markdown("### Mixture of Experts Approach")
        
        with gr.Row():
            with gr.Column():
                modulation = gr.Dropdown(
                    choices=["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "GFSK"],
                    value="QPSK",
                    label="Modulation Type"
                )
                snr = gr.Slider(
                    minimum=-20, maximum=20, value=10,
                    label="SNR (dB)"
                )
                generate_btn = gr.Button("Generate & Classify", variant="primary")
            
            with gr.Column():
                output_plot = gr.Plot(label="Signal Visualization")
                
                with gr.Row():
                    prediction = gr.Textbox(label="Predicted Modulation")
                    confidence = gr.Textbox(label="Confidence")
                    expert = gr.Textbox(label="Expert Used")
        
        generate_btn.click(
            fn=process_signal,
            inputs=[modulation, snr],
            outputs=[output_plot, prediction, confidence, expert]
        )
        
        gr.Markdown("""
        ## How It Works
        1. **SNR Estimation**: The model first estimates the SNR of the input signal
        2. **Expert Routing**: Based on estimated SNR, routes to appropriate expert
        3. **Classification**: Specialized expert classifies the modulation
        4. **Result**: Returns prediction with confidence and which expert was used
        
        ### Expert Networks:
        - **Low SNR Expert**: Optimized for noisy signals (-20 to 0 dB)
        - **Mid SNR Expert**: Balanced for moderate noise (0 to 10 dB)
        - **High SNR Expert**: High precision for clean signals (10 to 20 dB)
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)  # Set share=False for local only