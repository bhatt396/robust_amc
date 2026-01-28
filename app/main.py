import os
import numpy as np
import torch
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
from pathlib import Path

# Add parent directory to path to import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.moe_amc import MoEAMC
from config.config import Config
from data.generator import SignalGenerator
from data.dataset import load_rml_data

app = FastAPI(title="Robust AMC API")

# Ensure directories exist
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Create a placeholder file in static directory
with open(os.path.join(STATIC_DIR, '.gitkeep'), 'w') as f:
    pass

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Mount results directory to visualize training performance
RESULTS_DIR = os.path.join(Config.BASE_DIR, 'results')
if os.path.exists(RESULTS_DIR):
    app.mount("/results_assets", StaticFiles(directory=RESULTS_DIR), name="results_assets")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load configuration and model
config = Config()
model = None
generator = None
rml_signals = None
rml_labels = None
rml_snrs = None
rml_mod_names = None

class PredictionRequest(BaseModel):
    iq_data: list[list[float]]
    snr: float = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    expert_used: str
    snr_estimate: float

@app.on_event("startup")
async def load_model():
    """Load the trained model and signal generator"""
    global model, generator, rml_signals, rml_labels, rml_snrs, rml_mod_names
    
    try:
        # Initialize model
        model = MoEAMC(
            num_experts=config.NUM_EXPERTS,
            num_classes=config.NUM_CLASSES,
            input_channels=2,
            expert_filters=config.EXPERT_CNN_FILTERS,
            gating_mode='soft'
        ).to(config.DEVICE)
        
        # Get absolute path to model checkpoint
        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
        model_path = os.path.join(model_dir, 'moe_amc_best.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please make sure to train the model first by running 'python train.py'"
            )
            
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
        
        # Initialize signal generator
        generator = SignalGenerator(
            samples_per_symbol=config.SAMPLES_PER_SYMBOL,
            num_symbols=config.NUM_SYMBOLS
        )
        
        # Load RML data for sampling if needed
        if config.DATA_SOURCE == 'rml':
            print(f"Loading RML dataset for sampling from {config.RML_FILE}...")
            rml_signals, rml_labels, rml_snrs, rml_mod_names = load_rml_data(config.RML_FILE)
            
        print("Model and data sources loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTo train the model, run:")
        print("1. python train.py")
        print("2. Then start the API with: python -m uvicorn app.main:app --reload")
        raise e

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    # Check for result images
    performance_plots = []
    results_dir = Path(Config.RESULTS_PATH)
    if results_dir.exists():
        performance_plots = [f.name for f in results_dir.glob("*.png")]
        
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "modulations": config.MODULATIONS,
        "performance_plots": performance_plots
    })

@app.post("/predict", response_model=PredictionResponse)
async def predict_modulation(request: PredictionRequest):
    """API endpoint for modulation prediction"""
    if not request.iq_data:
        raise HTTPException(status_code=400, detail="Empty IQ data")
        
    iq_data = np.array(request.iq_data, dtype=np.float32)
    
    # Ensure we have the correct length
    target_length = config.SAMPLE_LENGTH
    current_length = len(iq_data)
    
    if current_length < target_length:
        padding = np.zeros((target_length - current_length, 2), dtype=np.float32)
        iq_data = np.vstack([iq_data, padding])
    elif current_length > target_length:
        iq_data = iq_data[:target_length]
    
    # Unit power normalization
    power = np.mean(iq_data[:, 0]**2 + iq_data[:, 1]**2)
    if power > 0:
        iq_data = iq_data / np.sqrt(power)
    
    # Convert to tensor: shape (1, 2, L)
    iq_tensor = torch.tensor(np.stack([iq_data[:, 0], iq_data[:, 1]], axis=0), 
                            dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs, expert_outputs, gating_weights, snr_probs = model(iq_tensor, return_expert_outputs=True)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        
        expert_idx = torch.argmax(gating_weights[0]).item()
        expert_used = config.SNR_BINS[expert_idx]
        
        # Consistent SNR estimation logic
        snr_centers = [
            (config.SNR_LOW[0] + config.SNR_LOW[1]) / 2,
            (config.SNR_MID[0] + config.SNR_MID[1]) / 2,
            (config.SNR_HIGH[0] + config.SNR_HIGH[1]) / 2
        ]
        estimated_snr = 0.0
        for i, center in enumerate(snr_centers):
            prob = snr_probs[0, i].item()
            estimated_snr += prob * center
        
        return {
            "prediction": config.MODULATIONS[pred.item()],
            "confidence": confidence.item(),
            "expert_used": expert_used,
            "snr_estimate": estimated_snr
        }

@app.post("/generate")
async def generate_signal(request: Request):
    """Generate a signal with the specified modulation and SNR"""
    try:
        # Parse form data
        form_data = await request.form()
        modulation = form_data.get("modulation")
        snr_str = form_data.get("snr", "0.0")
        try:
            snr = float(snr_str)
        except ValueError:
            snr = 0.0
            
        print(f"Received request - Modulation: {modulation}, SNR: {snr}")
        
        if modulation not in config.MODULATIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid modulation. Must be one of: {config.MODULATIONS}"}
            )
        
        # Generate or sample signal
        if config.DATA_SOURCE == 'rml' and rml_signals is not None:
            # Sample from RML dataset
            mod_idx = config.MODULATIONS.index(modulation)
            # Find closest SNR in dataset
            available_snrs = np.unique(rml_snrs)
            closest_snr = available_snrs[np.argmin(np.abs(available_snrs - snr))]
            
            # Find indices for this modulation and SNR
            mask = (rml_labels == mod_idx) & (rml_snrs == closest_snr)
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                # Fallback to any SNR for this modulation
                indices = np.where(rml_labels == mod_idx)[0]
                
            if len(indices) == 0:
                raise ValueError(f"No samples found for modulation {modulation}")
                
            idx = np.random.choice(indices)
            signal = rml_signals[idx]
        else:
            # Generate using generator (only works for supported modulations)
            try:
                signal = generator.generate_signal(modulation)
                if snr is not None:
                    signal = generator.add_awgn(signal, snr)
            except KeyError:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Generation not supported for {modulation} in 'generated' mode. Use 'rml' data source."}
                )
        
        # Normalize
        signal = signal / np.max(np.abs(signal) + 1e-10)  # Add small value to avoid division by zero
        
        # Convert complex to list of [real, imag] pairs
        iq_data = [[s.real, s.imag] for s in signal]
        
        return {
            "iq_data": iq_data,
            "modulation": modulation,
            "snr": snr
        }
        
    except Exception as e:
        print(f"Error in generate_signal: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error generating signal: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
