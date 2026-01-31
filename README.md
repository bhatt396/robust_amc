# SNR-Adaptive Automatic Modulation Classification using Mixture of Experts

Implementation of an SNR-adaptive Automatic Modulation Classification (AMC) system using a Mixture of Experts (MoE) architecture in PyTorch.

## Project Overview

This project implements a novel approach to automatic modulation classification that adapts to different Signal-to-Noise Ratio (SNR) conditions. Instead of using a single classifier for all SNR ranges, the system employs multiple expert networks, each specialized for specific SNR regimes (low, mid, high), coordinated by an SNR estimator and gating network.

## Features

- **SNR Estimation**: Lightweight CNN-based SNR estimator
- **Expert Networks**: Multiple specialized CNN classifiers for different SNR ranges
- **Adaptive Gating**: Soft and hard gating mechanisms for expert selection
- **8 Modulation Types**: BPSK, QPSK, 8PSK, QAM16, QAM64, GFSK, CPFSK, PAM4
- **Channel Models**: AWGN, Rayleigh, and Rician fading
- **Comprehensive Evaluation**: Accuracy vs SNR, per-modulation analysis, confusion matrices

## Project Structure

```
SNR-Adaptive-AMC/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Dataset classes
│   ├── generator.py        # Signal generation
│   └── preprocessing.py    # Preprocessing utilities
├── models/
│   ├── __init__.py
│   ├── snr_estimator.py   # SNR estimation network
│   ├── expert_networks.py  # Expert CNN classifiers
│   ├── gating_network.py   # Gating mechanisms
│   └── moe_amc.py          # Complete MoE system
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Plotting utilities
├── config/
│   ├── __init__.py
│   └── config.py           # Configuration parameters
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SaurabPoudel/SNR-Adaptive-AMC.git
cd SNR-Adaptive-AMC
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

The training process consists of three stages:

1. **Train SNR Estimator**:
```python
python train.py
```

This will:
- Generate a dataset of I/Q samples
- Train the SNR estimator to classify SNR bins
- Train individual expert networks for each SNR range
- Fine-tune the complete MoE system end-to-end

### Evaluation

Evaluate the trained model:
```python
python evaluate.py
```

This will generate:
- Overall classification accuracy
- Accuracy vs SNR curves
- Per-modulation accuracy
- Confusion matrices
- Expert contribution analysis

### Custom Configuration

Modify `config/config.py` to adjust:
- Modulation types
- SNR ranges
- Model architectures
- Training hyperparameters
- Data generation parameters

## Model Architecture

### System Overview
```
Input I/Q Signal → SNR Estimator → Gating Network → {Expert_Low, Expert_Mid, Expert_High}
                                                    ↓
                                            Weighted Combination
                                                    ↓
                                         Final Modulation Prediction
```

### Components

1. **SNR Estimator**: CNN-based network that estimates SNR bin probabilities
2. **Expert Networks**: Specialized CNN classifiers for different SNR ranges
3. **Gating Network**: Produces weights for combining expert predictions
4. **MoE Fusion**: Combines expert outputs using soft or hard gating

## Results

Expected performance:
- **Low SNR (-10 to 0 dB)**: ~60-80% accuracy
- **Mid SNR (0 to 10 dB)**: ~80-95% accuracy
- **High SNR (10 to 20 dB)**: ~95-99% accuracy
- **Overall**: 5-10% improvement over baseline single-model approach

### Dataset Configuration

The project supports two data sources, which can be toggled in `config/config.py`:

1.  **Synthetic Data (`DATA_SOURCE = 'generated'`)**: Uses the built-in `SignalGenerator` to create I/Q samples with configurable modulations and channel effects.
2.  **RadioML 2016.10a (`DATA_SOURCE = 'rml'`)**: Uses the standard `RML2016.10a_dict.pkl` dataset.

To use the RadioML dataset:
1.  Place `RML2016.10a_dict.pkl` in the `data/` directory.
2.  Set `DATA_SOURCE = 'rml'` in `config/config.py`.
3.  The system will automatically adjust modulation classes (11 for RML) and sample lengths (128).

## Training Tips

1. **Pre-train components**: Train SNR estimator and experts separately before end-to-end training
2. **Data balancing**: Ensure balanced representation across SNR ranges and modulation types
3. **Learning rate**: Use lower learning rate for fine-tuning the complete system
4. **Regularization**: Apply dropout and weight decay to prevent overfitting

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{snr-adaptive-amc,
  author = {Bhaskar Bhatt, Dharma Raj Thapa, Saurab Poudel, Sujan Bimali},
  title = {SNR-Adaptive Automatic Modulation Classification using Mixture of Experts},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/SaurabPoudel/SNR-Adaptive-AMC}
}
```

## License

MIT License

## Contributors

- Bhaskar Bhatt (079BEI013)
- Dharma Raj Thapa (079BEI017)
- Saurab Poudel (079BEI036)
- Sujan Bimali (079BEI044)

## Acknowledgments

This project was developed as part of the undergraduate final year project at Pulchowk Campus, IOE, Tribhuvan University.