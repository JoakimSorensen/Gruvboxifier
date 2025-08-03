# Gruvbox Image Converter

GPU-accelerated tool to convert any image to the [Gruvbox color palette](https://github.com/morhetz/gruvbox) with perceptually-accurate color matching.

## Features

- **GPU Acceleration**: PyTorch/CuPy support for speedup
- **Perceptual Color Matching**: LAB color space tries to prevent gray over-selection
- **Extended Palette**: Additional intermediate colors for better results
- **Multiple Algorithms**: LAB, HSV, weighted RGB, and gray-biased methods

## Installation

```bash
# GPU acceleration (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# or
pip install cupy-cuda12x

# Required dependencies
pip install pillow numpy
```

## Usage

```bash
# Best quality (recommended)
python gruvbox_gpu_converter.py input.jpg output.jpg

# Reduce gray over-selection
python gruvbox_gpu_converter.py input.jpg output.jpg --color-method gray_biased

# Preserve hue better
python gruvbox_gpu_converter.py input.jpg output.jpg --color-method hsv

# Use original Gruvbox palette only
python gruvbox_gpu_converter.py input.jpg output.jpg --original-palette
```

## Examples

| Original | Gruvbox |
|----------|---------|
| ![Original 1](./example_images/grass.JPG) | ![Converted 1](./example_images/grass_gruvbox.JPG) |
| ![Original 2](./example_images/furnace_monster.JPG) | ![Converted 2](./example_images/furnace_monster_gruvbox.JPG) |

## How It Works

The converter uses **LAB color space** instead of RGB for perceptually-uniform color matching. This means colors that look similar to humans will be matched together, trying to prevent the common issue where mid-tones map to gray.

**Color Methods:**
- `lab` (default) - Should be most perceptually accurate, best for reducing gray mapping
- `hsv` - Good for preserving color character and hue, can wield some strange results
- `gray_biased` - Applies penalty to gray selection
- `weighted_rgb` - Emphasizes luminance differences
- `standard` - Basic RGB Euclidean distance

## Performance

- **Small images** (1MP): ~1-5 seconds
- **Large images** (25MP): ~30-120 seconds with GPU
- **CPU fallback**: Still 2-5x faster than naive approaches

Requires CUDA-compatible GPU for best performance. Falls back to optimized CPU operations if GPU unavailable.
