# 🐸 Coqui TTS Web Interface

Enhanced Coqui TTS with comprehensive web interface, voice cloning, and Docker support for containerized deployment.

## Features

### 🎙️ Advanced Text-to-Speech
- **Multi-language support**: 16+ languages including English, Spanish, French, German, and more
- **Voice cloning**: Upload audio samples to clone any voice
- **Real-time synthesis**: Fast audio generation with streaming support
- **Advanced controls**: Speed, pitch, and volume adjustment
- **Model selection**: Switch between different TTS models

### 🎭 Voice Cloning System
- **Upload audio samples** (WAV, MP3, FLAC)
- **Clone voices** with high accuracy using XTTS models
- **Save voice profiles** for reuse
- **Voice library management** with easy selection

### 📁 File Management
- **Download all generated audio** files
- **Audio library** with playback controls
- **Persistent storage** across container restarts
- **File organization** by date and type

### 🔧 Technical Features
- **Docker containerized** for easy deployment
- **REST API** for integration with other services
- **Real-time web interface** with modern UI
- **GPU acceleration** support (CUDA)
- **Health monitoring** and status indicators

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jyoung2000/j-coqui-ClaudeCode.git
   cd j-coqui-ClaudeCode
   ```

2. **Build and run**:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. **Access the web interface**:
   - Open your browser to `http://localhost:2201`
   - Start generating speech and cloning voices!

### Manual Docker Build

```bash
# Build the container
docker build -f Dockerfile.web -t coqui-tts-web .

# Run with volume mounts for data persistence
docker run -d \
  -p 2201:2201 \
  -v $(pwd)/data/models:/app/models \
  -v $(pwd)/data/voices:/app/voices \
  -v $(pwd)/data/outputs:/app/outputs \
  -v $(pwd)/data/uploads:/app/uploads \
  --name coqui-tts-web \
  coqui-tts-web
```

## API Endpoints

### Text-to-Speech
```bash
POST /api/tts
{
  "text": "Hello world!",
  "voice_profile": "my_voice",
  "language": "en"
}
```

### Voice Cloning
```bash
POST /api/clone_voice
# Form data with audio file and voice_name
```

### Voice Management
```bash
GET /api/voices                    # List all voices
DELETE /api/delete_voice/{name}    # Delete a voice
```

### File Management
```bash
GET /api/outputs                   # List generated files
GET /api/download/{filename}       # Download a file
```

### System Status
```bash
GET /health                        # Health check
GET /api/models                    # Available models
```

## Directory Structure

```
├── web_server/                    # Enhanced web application
│   ├── app.py                    # Main Flask application
│   ├── templates/                # HTML templates
│   └── static/                   # CSS, JS, assets
├── TTS/                          # Original Coqui TTS library
├── data/                         # Persistent data (created by container)
│   ├── models/                   # Downloaded TTS models
│   ├── voices/                   # Cloned voice profiles
│   ├── outputs/                  # Generated audio files
│   └── uploads/                  # Temporary uploads
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile.web               # Optimized production Dockerfile
└── README_WEB.md                # This file
```

## Voice Cloning Guide

1. **Prepare audio sample**:
   - 10-30 seconds of clear speech
   - Minimal background noise
   - Single speaker
   - Supported formats: WAV, MP3, FLAC

2. **Upload and clone**:
   - Use the web interface upload area
   - Enter a memorable voice name
   - Click "Clone Voice"

3. **Use cloned voice**:
   - Select from the voice dropdown
   - Enter text and generate speech
   - Download the result

## Configuration

### Environment Variables
- `PORT`: Web server port (default: 2201)
- `TTS_MODELS_DIR`: Model storage directory
- `TTS_VOICES_DIR`: Voice profiles directory
- `TTS_OUTPUTS_DIR`: Generated files directory
- `TTS_UPLOADS_DIR`: Upload temporary directory

### Volume Mounts
- `/app/models`: TTS model cache
- `/app/voices`: Voice profiles and audio samples
- `/app/outputs`: Generated audio files
- `/app/uploads`: Temporary upload storage

## GPU Support

The container supports NVIDIA GPU acceleration:

```bash
# For GPU support
docker-compose up -d
# or
docker run --gpus all -p 2201:2201 coqui-tts-web
```

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run development server
cd web_server
python app.py
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Submit a pull request

## Troubleshooting

### Common Issues

1. **Container won't start**:
   - Check Docker logs: `docker-compose logs`
   - Ensure port 2201 is available

2. **No audio output**:
   - Check browser console for errors
   - Verify file permissions in output directory

3. **Voice cloning fails**:
   - Ensure audio file is under 10MB
   - Use supported audio formats
   - Check audio quality (clear speech, no background noise)

4. **GPU not detected**:
   - Install NVIDIA Docker runtime
   - Verify GPU availability: `nvidia-smi`

### Performance Tips
- Use GPU for faster synthesis (requires NVIDIA GPU + Docker)
- Keep voice samples under 30 seconds for optimal cloning
- Clear old files periodically to save disk space

## License

This project is based on Coqui TTS, which is released under the Mozilla Public License 2.0. See the original repository for license details.

## Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support and questions
- **Documentation**: See Coqui TTS official docs for model details

---

Made with ❤️ using Coqui TTS and enhanced with a modern web interface for the community.