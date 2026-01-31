# Robust AMC API

A FastAPI-based web application for Automatic Modulation Classification (AMC) using a Mixture of Experts (MoE) model.

## Features

- Real-time signal generation with different modulation schemes
- Interactive visualization of I/Q signals and constellation diagrams
- Model inference with confidence scores and expert usage information
- Responsive web interface

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements-api.txt
   ```

2. Make sure you have the trained model weights at `checkpoints/moe_amc_best.pth`

## Running the Application

1. Start the FastAPI server:
   ```bash
   cd app
   uvicorn main:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Classify a signal (expects JSON with `iq_data` array)
- `POST /generate`: Generate a signal with specified modulation and SNR

## Project Structure

- `main.py`: FastAPI application and API endpoints
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JS, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
