# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Coqui TTS** - a comprehensive text-to-speech library built on PyTorch that supports training and inference of multiple TTS models. The project includes voice cloning, multi-speaker synthesis, and over 1100 language support.

## Key Build & Development Commands

### Installation & Setup
```bash
# Install system dependencies (Ubuntu/Debian)
make system-deps

# Install development dependencies  
make dev-deps

# Install TTS for development
make install
# or: pip install -e .[all]
```

### Testing
```bash
# Run all tests
make test

# Run specific test suites
make test_tts           # TTS model tests
make test_vocoder       # Vocoder tests
make test_aux           # Auxiliary tests
make test_text          # Text processing tests
make inference_tests    # Inference tests
```

### Code Quality
```bash
# Format code
make style              # Run black and isort

# Lint code
make lint               # Run pylint, black --check, isort --check
```

### Documentation
```bash
# Build documentation
make docs
# or: make build-docs
```

## Architecture Overview

### Core Components

**TTS Pipeline Architecture:**
- **TTS Models** (`TTS/tts/models/`): Neural TTS models (Tacotron, VITS, XTTS, etc.)
- **Vocoders** (`TTS/vocoder/`): Neural vocoders (HiFiGAN, WaveGrad, etc.)
- **Speaker Encoders** (`TTS/encoder/`): Models for speaker embeddings and voice cloning
- **Text Processing** (`TTS/tts/utils/text/`): Multi-language text normalization and phonemization
- **Audio Processing** (`TTS/utils/audio/`): Audio preprocessing and transforms

**Model Configuration System:**
- All models use configuration classes in `TTS/tts/configs/` that extend `BaseTTSConfig`
- Configurations are JSON-serializable and handle model hyperparameters
- Each model type has its own config class (e.g., `XttsConfig`, `VitsConfig`)

**Training Framework:**
- Built on the `trainer` library for distributed training
- Training scripts in `TTS/bin/` handle data loading, model setup, and training loops
- Supports multi-GPU training, gradient accumulation, and checkpointing

### Key Training Scripts

Located in `TTS/bin/`:
- `train_tts.py`: Main TTS model training
- `train_vocoder.py`: Vocoder training  
- `train_encoder.py`: Speaker encoder training
- `synthesize.py`: Comprehensive synthesis tool with CLI
- `compute_embeddings.py`: Generate speaker embeddings for voice cloning
- `extract_tts_spectrograms.py`: Extract spectrograms for vocoder training

### Model Management

**ModelManager** (`TTS/utils/manage.py`):
- Downloads and manages pre-trained models
- Models stored in `~/.local/share/tts/` 
- Handles model versioning and licensing
- Provides unified interface for model loading

**API Interface** (`TTS/api.py`):
- High-level Python API for TTS synthesis
- Supports voice cloning, multi-speaker synthesis, and voice conversion
- Automatically handles model loading and audio processing

### Web Server

**Flask Server** (`TTS/server/server.py`):
- Basic web interface for TTS synthesis
- REST API endpoint at `/api/tts`
- Supports multi-speaker and multi-language models
- Simple HTML interface in `TTS/server/templates/`

## Development Patterns

### Model Implementation
- All TTS models inherit from `BaseTTS` in `TTS/tts/models/base_tts.py`
- Models implement `forward()`, `inference()`, and `init_from_config()` methods
- Configuration classes define model hyperparameters and are used for serialization

### Dataset Handling
- Dataset formatters in `TTS/tts/datasets/formatters.py` handle different dataset formats
- Audio preprocessing handled by `AudioProcessor` class
- Text processing varies by language and uses phonemizers in `TTS/tts/utils/text/phonemizers/`

### Training Configuration
- Training uses JSON config files that specify model, dataset, and training parameters
- Recipes in `recipes/` directory provide example configurations for different datasets
- Supports resume training, distributed training, and tensorboard logging

## Voice Cloning & Multi-Speaker Support

The project supports two main approaches for voice cloning:
1. **Speaker Encoders**: Generate speaker embeddings from reference audio
2. **Multi-Speaker Models**: Train on multiple speakers with speaker IDs

Key components:
- `compute_embeddings.py`: Generate speaker embeddings
- XTTS models: Production-ready voice cloning with real-time inference
- YourTTS: Multi-language voice cloning model

## Docker Support

- `Dockerfile`: CUDA-enabled container for GPU inference
- `dockerfiles/Dockerfile.dev`: Development container
- Container runs `tts` command by default with `--help`

## Important Notes

- Python 3.9-3.12 supported
- CUDA support via PyTorch for GPU acceleration
- Models are downloaded automatically on first use
- Configuration files are essential for training - see `recipes/` for examples
- The project uses `coqpit` for configuration management and `trainer` for training