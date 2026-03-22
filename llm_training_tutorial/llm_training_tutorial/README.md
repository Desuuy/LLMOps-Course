# LLM Training Tutorial Environment

This is a self-contained tutorial environment designed to provide students with a hands-on experience in fine-tuning the Gemma-3-1B-IT model for tool calling. The setup emphasizes modern best practices by separating infrastructure services into containers and executing model training locally within an optimized Conda environment.

## Overview

- **Infrastructure**: Uses Docker Compose to independently host `MinIO` (S3-compatible object storage) and `MLflow` (experiment tracking).
- **Environment**: Relies on a Conda environment inside `tmux` to allow uninterrupted, long-running training tasks directly on the VM.
- **Optimized Training**: Leverages Unsloth, TRL, and LoRA optimizations for faster training and validation.

## Prerequisites
- **Docker & Docker Compose**: Ensure Docker is installed to run the tracking infrastructure.
- **Conda**: Required to build an isolated, GPU-compatible Python 3.10 environment.
- **tmux**: Ensures your terminal session doesn't die when you disconnect from the VM.

## Step-by-Step Guide

### 1. Launch the Infrastructure
We begin by spinning up MinIO and MLflow. 
```bash
# Navigate to the tutorial directory (where docker-compose.yml lives)
cd llm_training_tutorial

# Start background services
docker compose up -d
```
You can verify the services are running:
- **MinIO Console**: `http://localhost:9101` (Creds: `minioadmin` / `minioadmin123` by default as per `.env`)
- **MLflow UI**: `http://localhost:5000`

### 2. Set up the Conda Environment
Running the training inside `tmux` is highly recommended for stability.

```bash
# Open a new tmux session
tmux new -s training_session

# Navigate to the train folder
cd train

# Create and activate the conda environment
conda create -n gemma_train python=3.10 -y
conda activate gemma_train

# Install all dependencies using the provided requirements.txt
pip install -r requirements.txt
```

### 3. Pre-Download the Model
To save time and prevent repetitive downloads during script restarts, use the provided downloader.
```bash
# Navigate back to the tutorial root directory to run the downloader
cd ..
python download_model.py
```
This extracts `google/gemma-3-1b-it` into the local `models` folder.

### 4. Run the Training Script
We are now ready to train the model. Ensure you run this from within the `train` directory so relative paths resolve correctly. The script automatically reads properties from `config.json`.

```bash
cd train
python train.py --config config.json
```

### What Happens During Training?
1. The script loads the base model from the local `models/gemma-3-1b-it` directory.
2. It parses the dataset and structures ChatML prompts for Tool Calling.
3. Automatically runs a rapid pre-training evaluation step using cached inference logic to generate baseline accuracy, logged immediately to MLflow.
4. Performs LoRA (Low-Rank Adaptation) PEFT fine-tuning with hyper-parameters defined in `config.json`.
5. Re-evaluates using the best LoRA checkpoints and exports the accuracy metrics to contrast pre/post training behaviors.
6. The LoRA adapter is securely serialized and pushed directly to MinIO, accessible within your MLflow dashboard.

### 5. Merge Model (Optional for Serving)
If you wish to serve the model with engines like `vLLM` or `SGLang`, it's usually preferred to merge the base model with the new LoRA weights. We provide a script to download a specific trained LoRA adapter from your local S3 MinIO storage, mount it on the base model, and bake them into a single local artifact.

In the MLflow UI (`http://localhost:5000`), copy the **Run ID** of your successful fine-tuning experiment, then run:

```bash
cd train
python merge_model.py --run_id YOUR_RUN_ID
```
The script outputs the merged 16-bit model into `../models/gemma-3-1b-it-merged`.

Access the MLflow UI (`http://localhost:5000`) anytime to monitor hardware system metrics, training loss, and your stored adapters.
