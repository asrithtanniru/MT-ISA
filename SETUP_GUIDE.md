# MT-ISA Implementation Setup Guide

## Prerequisites

### 1. Python Environment
```bash
python --version  # 3.9+
pip install --upgrade pip
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers==4.35.0
pip install datasets
pip install pandas numpy scipy
pip install ollama
pip install accelerate
pip install scikit-learn
pip install tqdm
```

### 3. Ollama Setup (For Local LLM)

#### Installation
```bash
# Download from: https://ollama.ai/

# Or using package managers:
# macOS
brew install ollama

# Linux
curl https://ollama.ai/install.sh | sh

# Windows
# Download installer from ollama.ai
```

#### Start Ollama Service
```bash
# Start ollama in background
ollama serve

# In another terminal, pull a local LLM model
ollama pull mistral          # Faster, smaller (7B)
# OR
ollama pull neural-chat      # Better quality (7B)
# OR
ollama pull llama2          # Larger (7B-70B variants)
```

#### Test Ollama Connection
```bash
ollama list  # Should show your downloaded models

# Test API
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Hello, world!",
  "stream": false
}'
```

### 4. Dataset Preparation

#### Option A: Download SemEval-2014
```bash
# Create data directory
mkdir -p data/semeval2014

# Download from: https://www.aclweb.org/anthology/S14-2004/
# Or use direct links (check paper references)

# Extract to: data/semeval2014/
# Expected files:
# - Restaurants_Train.xml
# - Restaurants_Test.xml
# - Laptops_Train.xml
# - Laptops_Test.xml
```

#### Option B: Use Our Sample Data Script
```python
# See data_loader.py script for conversion utilities
python data_loader.py --download semeval2014
```

### 5. GPU Setup (Optional but Recommended)

#### Check GPU
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### CUDA Installation
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For M1/M2 Macs
# PyTorch automatically uses MPS (Metal Performance Shaders)
```

---

## Project Structure

```
mt-isa-implementation/
├── data/
│   ├── semeval2014/
│   │   ├── Restaurants_Train.xml
│   │   ├── Restaurants_Test.xml
│   │   ├── Laptops_Train.xml
│   │   └── Laptops_Test.xml
│   └── processed/
│       ├── train.json
│       └── test.json
│
├── models/
│   ├── checkpoint-best/
│   │   └── pytorch_model.bin
│   └── outputs/
│       └── metrics.json
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load & parse datasets
│   ├── ollama_client.py         # LLM integration
│   ├── auxiliary_generator.py   # Phase 1 (self-refine)
│   ├── model.py               # Flan-T5 + D-AWL + T-AWL
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation metrics
│   └── utils.py               # Helper functions
│
├── config.yaml                 # Configuration
├── requirements.txt
├── train.py                   # Main entry point
├── evaluate.py                # Evaluation entry point
└── README.md
```

---

## Configuration Example

Create `config.yaml`:
```yaml
# Model Configuration
model:
  backbone: "google/flan-t5-small"  # or base, large, xl, xxl
  pretrained: true

# Data Configuration
data:
  dataset_name: "semeval2014"
  domain: "restaurant"  # or "laptop"
  split_type: "implicit"  # or "all", "explicit"
  train_path: "data/semeval2014/Restaurants_Train.xml"
  test_path: "data/semeval2014/Restaurants_Test.xml"

# Ollama LLM Configuration
ollama:
  model: "mistral"  # Change to your model
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 200

# Auxiliary Task Generation
auxiliary:
  max_refinement_epochs: 10
  temperature: 0.7
  confidence_threshold: 0.5  # Minimum confidence to use

# D-AWL Configuration
d_awl:
  strategy: "input"  # "input", "output", or "input_output"
  
# T-AWL Configuration
t_awl:
  loss_fn: "alf2"  # "alf1" or "alf2"
  uncertainty_init: 1.0

# Training Configuration
training:
  batch_size: 32
  learning_rate: 1e-5
  num_epochs: 20
  warmup_steps: 500
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  early_stopping_patience: 10
  seed: 42

# Logging
logging:
  log_interval: 50
  save_interval: 500
  output_dir: "models/outputs/"
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure Ollama is running
ollama serve  # In one terminal

# 3. Prepare data
python src/data_loader.py --dataset semeval2014

# 4. Generate auxiliary data
python src/auxiliary_generator.py --config config.yaml

# 5. Train model
python train.py --config config.yaml

# 6. Evaluate
python evaluate.py --model models/checkpoint-best/ --config config.yaml
```

---

## Troubleshooting

### Ollama Connection Error
```
Error: Failed to connect to Ollama at http://localhost:11434

Solution:
1. Ensure ollama serve is running
2. Check: curl http://localhost:11434/api/tags
3. Verify model is downloaded: ollama list
```

### Out of Memory
```
If GPU runs out of memory:
1. Reduce batch_size in config.yaml
2. Use smaller model: "google/flan-t5-small"
3. Enable gradient_accumulation_steps
4. Use CPU: Set CUDA_VISIBLE_DEVICES=""
```

### Slow Ollama Inference
```
If LLM generation is very slow:
1. Use faster model: mistral instead of llama2
2. Reduce max_tokens
3. Ensure Ollama has enough RAM
4. Check: ollama ps (shows running models)
```

---

## Next Steps

1. Follow the **Data Preparation** section
2. Run **Quick Start** commands
3. Check training progress in logs
4. Evaluate on test set

See individual script files (data_loader.py, train.py, etc.) for detailed documentation.
