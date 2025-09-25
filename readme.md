# Genshinfy v2 - AI Style Transfer Project

This project contains AI-powered style transfer tools for generating an anime-style avatar of your face from Genshin Impact character references (see [Quick Start](#-quick-start)).   
You can also just use AnythingV4 to generate avatars based off any Genshin Impact character you want (see [Detailed Usage](#-detailed-usage))

## Gradio Demo
Available [here](https://42cummer-genshinfyv2.hf.space/).

## 🎯 Project Structure

```
GenshinfyV2/
├── AnythingV4/                    # Style transfer using AnythingV4 model
│   ├── styletransfer.py          # Main Python script
│   ├── styletransfer.sh          # Shell script runner
│   ├── styletransfer.ipynb       # Jupyter notebook version
│   └── Avatar_like_*.png         # Generated images
├── DINOv2/                       # DINO Image-Image Similarity Matching
│   ├── dino.ipynb               # DINO notebook
│   ├── dino.py                  # DINO Python script
│   └── dino.sh                  # DINO shell script
├── GenshinCharacters/           # Character reference images (as of Genshin Impact version 6.0)
│   ├── Fischl.png
│   ├── Diluc.png
│   └── ... (100+ characters)
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project directory
cd GenshinfyV2

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run the Pipeline!!
```bash
# Navigate to DINOv2 directory
cd DINOv2

# Run image-image matching
./pipeline.sh <path_to_your_image> -t <optional_tags>
./pipeline.sh test_image.png -t "brown hair, red eyes"
```


## 📖 Detailed Usage

### Style Transfer Script (`styletransfer.py`)

The main script generates Avatar-style portraits based on Genshin Impact character images using the AnythingV4 diffusion model.

#### Features:
- **Automatic character extraction**: Extracts character name from file path
- **Interactive prompts**: Allows custom tags for generation
- **Memory management**: Automatically clears GPU/CPU memory after generation
- **Error handling**: Validates input files and provides clear error messages

#### Usage Options:

**Option 1: Using the shell script (recommended)**
```bash
cd AnythingV4
./styletransfer.sh ../GenshinCharacters/[CharacterName].png
```

**Option 2: Direct Python execution**
```bash
cd AnythingV4
python3 styletransfer.py ../GenshinCharacters/[CharacterName].png
```

#### Interactive Prompts:

When you run the script, you'll be prompted for optional tags:
```
Enter optional tags, separated by commas, or press enter to skip.
For example, 'blonde, green eyes'
```

Examples of good tags:
- `blonde, green eyes, elegant dress`
- `red hair, battle armor, determined`
- `blue hair, magical aura, serene`
- (Press Enter to skip and use default styling)

***Note: NSFW tags are allowed, as the safety checker has been disabled. Do not distribute generated NSFW images as this violates AnythingV4 usage policy***

#### Output:
- Generated images are saved as `Avatar_like_[character].png`
- Example: `Avatar_like_fischl.png`, `Avatar_like_jean.png`

## 🎨 Available Characters

The `GenshinCharacters/` directory contains 100+ character reference images including:

**Popular Characters:**
- Fischl, Diluc, Ayaka, Venti, Zhongli
- Hu Tao, Ganyu, Eula, Raiden Shogun
- Klee, Mona, Jean, Barbara
- And many more...

**Full List:**
Check the `GenshinCharacters/` directory for all available character images.

## Common Issues:

**1. Virtual Environment Not Found**
```bash
# Make sure you're in the project root directory
cd /path/to/GenshinfyV2
source .venv/bin/activate
```

**2. Permission Denied on Script**
```bash
chmod +x AnythingV4/styletransfer.sh
```

**3. File Not Found Error**
```bash
# Check if the character image exists
ls GenshinCharacters/Fischl.png
```

**4. Out of Memory Error**
- The script automatically manages memory
- Try closing other applications if issues persist
- Reduce `num_inference_steps` in the script for faster generation

**5. Model Download Issues**
- First run may take time to download models (~4GB)
- Ensure stable internet connection
- Models are cached locally after first download

## 📋 Requirements

### System Requirements:
- **Python**: 3.8+
- **GPU**: Strongly Recommended (CUDA/MPS support)
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ free space (for models)

### Python Dependencies:
See `requirements.txt` for complete list. Key packages:
- `torch`
- `diffusers`
- `transformers`
- `PIL`
- `accelerate`

## 📄 License

This project is for educational and research purposes. Please respect the licenses of the underlying models and datasets.

## 🙏 Acknowledgments

- **AnythingV4**: xyn-ai/anything-v4.0 model
- **Stable Diffusion**: Stability AI for the VAE
- **Genshin Impact**: miHoYo for character designs
- **Diffusers**: Hugging Face for the library

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure you're using the correct file paths
4. Check that the virtual environment is activated

---

**Happy generating! 🎨✨**
