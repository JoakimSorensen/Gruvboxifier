"""
MIT License

Copyright (c) 2025 Joakim Sorensen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
GPU-Accelerated Gruvbox Image Converter
Multiple GPU acceleration options using PyTorch, CuPy, and Numba

PERFORMANCE:
- GPU-accelerated LAB/HSV color space conversions
- Vectorized distance calculations using GPU tensor operations
- Extended Gruvbox palette with intermediate colors

RECOMMENDED USAGE for best results:
python gruvbox_gpu_converter.py image.jpg output.jpg --color-method lab --extended-palette

Uses perceptually-accurate LAB color space + extended palette to try to minimize gray mapping.
"""

from PIL import Image
import numpy as np
import argparse
import os
import time
import colorsys
from typing import Tuple, List, Optional


"""
Constants used in LAB conversion.
"""
GAMMA_THRESHOLD = 0.04045    # sRGB standard threshold
GAMMA_DIVISOR = 1.055        # sRGB standard divisor
GAMMA_EXPONENT = 2.4         # sRGB standard gamma exponent
LINEAR_DIVISOR = 12.92       # Linear section divisor
GAMMA_OFFSET = 0.055         # sRGB standard offset

D65_X = 0.95047    # X component of D65 illuminant
D65_Y = 1.00000    # Y component of D65 illuminant (always 1.0 by definition)
D65_Z = 1.08883    # Z component of D65 illuminant

LAB_THRESHOLD = 0.008856       # (6/29)³ - threshold for linear vs cube root
LAB_SLOPE = 7.787              # 1/3 * (29/6)² - slope of linear section
LAB_OFFSET = 16/116            # Offset for linear section
LAB_CUBE_ROOT = 1/3            # Cube root exponent

L_SCALE = 116          # Lightness scale factor
L_OFFSET = 16          # Lightness offset
A_SCALE = 500          # Green-red opponent scale
B_SCALE = 200          # Blue-yellow opponent scale

TRNS_MAT = [
    # X coefficients    Y coefficients    Z coefficients
    [0.4124564,        0.3575761,        0.1804375],     # Red contribution
    [0.2126729,        0.7151522,        0.0721750],     # Green contribution
    [0.0193339,        0.1191920,        0.9503041]      # Blue contribution
]


"""
Constants used for pallette.
TODO: Move to files and add more modes than gruvbox.
"""
# Gruvbox color palette
GRUVBOX_PALETTE = {
    'bg0_h': '#1d2021', 'bg0': '#282828', 'bg0_s': '#32302f', 'bg1': '#3c3836',
    'bg2': '#504945', 'bg3': '#665c54', 'bg4': '#7c6f64', 'fg4': '#a89984',
    'fg3': '#bdae93', 'fg2': '#d5c4a1', 'fg1': '#ebdbb2', 'fg0': '#fbf1c7',
    'red_dark': '#cc241d', 'red_light': '#fb4934', 'green_dark': '#98971a',
    'green_light': '#b8bb26', 'yellow_dark': '#d79921', 'yellow_light': '#fabd2f',
    'blue_dark': '#458588', 'blue_light': '#83a598', 'purple_dark': '#b16286',
    'purple_light': '#d3869b', 'aqua_dark': '#689d6a', 'aqua_light': '#8ec07c',
    'orange_dark': '#d65d0e', 'orange_light': '#fe8019', 'gray': '#928374',
}

# Extended palette to reduce gray over-selection
EXTENDED_GRUVBOX_PALETTE = {
    **GRUVBOX_PALETTE,
    'red_medium': '#e85d75', 'green_medium': '#a9b665', 'blue_medium': '#6d9fc7',
    'yellow_medium': '#e9b143', 'purple_medium': '#c18fcf', 'aqua_medium': '#7fb069',
    'orange_medium': '#f28534', 'brown': '#af3a03', 'dark_blue': '#2e3440',
    'dark_green': '#5f7c5f', 'dark_purple': '#7f5f7f', 'warm_gray': '#a89984',
    'cool_gray': '#7c7c7c'
}


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to LAB color space (perceptually uniform)."""
    # normalize
    r, g, b = [x / 255.0 for x in rgb]

    def gamma_correct(c):
        return ((c + GAMMA_OFFSET) / GAMMA_DIVISOR) ** GAMMA_EXPONENT if c > GAMMA_THRESHOLD else c / LINEAR_DIVISOR

    r, g, b = map(gamma_correct, [r, g, b])

    # Convert to XYZ
    x = r * TRNS_MAT[0][0] + g * TRNS_MAT[0][1] + b * TRNS_MAT[0][2]
    y = r * TRNS_MAT[1][0] + g * TRNS_MAT[1][1] + b * TRNS_MAT[1][2]
    z = r * TRNS_MAT[2][0] + g * TRNS_MAT[2][1] + b * TRNS_MAT[2][2]

    # Normalize by D65 illuminant
    x, y, z = x / D65_X, y / D65_Y, z / D65_Z

    def f_transform(t):
        return t**(LAB_CUBE_ROOT) if t > LAB_THRESHOLD else (LAB_SLOPE * t + LAB_OFFSET)

    fx, fy, fz = map(f_transform, [x, y, z])

    L = L_SCALE * fy - L_OFFSET
    a = A_SCALE * (fx - fy)
    b = B_SCALE * (fy - fz)

    return (L, a, b)


def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB to HSV color space."""
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hsv(r, g, b)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# Method PyTorch
def convert_with_pytorch(img_array: np.ndarray, color_method: str = 'lab',
                        use_extended_palette: bool = True, preserve_alpha: bool = True) -> np.ndarray:
    """Convert image using PyTorch GPU acceleration with advanced color matching."""
    try:
        import torch
        print(f"Using PyTorch GPU acceleration with {color_method} color matching")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        if device.type == 'cpu':
            print("Warning: CUDA not available, falling back to CPU")

        # Choose palette
        palette = EXTENDED_GRUVBOX_PALETTE if use_extended_palette else GRUVBOX_PALETTE

        h, w = img_array.shape[:2]
        rgb_data = torch.tensor(img_array[:, :, :3], dtype=torch.float32, device=device)

        # GPU-accelerated color space conversions
        if color_method == 'lab':
            print("Converting to LAB color space on GPU...")
            img_processed = rgb_to_lab_gpu(rgb_data)
            palette_processed = torch.stack([
                rgb_to_lab_gpu(torch.tensor(hex_to_rgb(color), dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
                for color in palette.values()
            ])

        elif color_method == 'hsv':
            print("Converting to HSV color space on GPU...")
            img_processed = rgb_to_hsv_gpu(rgb_data)
            palette_processed = torch.stack([
                rgb_to_hsv_gpu(torch.tensor(hex_to_rgb(color), dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
                for color in palette.values()
            ])

        elif color_method == 'weighted_rgb':
            # Apply RGB perceptual weights
            weights = torch.tensor([0.3, 0.59, 0.11], device=device)  # R, G, B weights
            img_processed = rgb_data * weights
            palette_processed = torch.stack([
                torch.tensor(hex_to_rgb(color), dtype=torch.float32, device=device) * weights
                for color in palette.values()
            ])

        else:  # standard RGB
            img_processed = rgb_data
            palette_processed = torch.stack([
                torch.tensor(hex_to_rgb(color), dtype=torch.float32, device=device)
                for color in palette.values()
            ])

        # Reshape for vectorized distance calculation
        pixels = img_processed.reshape(-1, 3)

        # Vectorized distance calculation on GPU
        distances = torch.cdist(pixels.unsqueeze(0), palette_processed.unsqueeze(0)).squeeze(0)

        # Apply method-specific adjustments
        if color_method == 'gray_biased' and 'gray' in palette:
            gray_idx = list(palette.keys()).index('gray')
            distances[:, gray_idx] *= 1.5  # Bias against gray

        # Find closest colors
        closest_indices = torch.argmin(distances, dim=1)

        # Map back to RGB colors for output
        rgb_palette = torch.stack([
            torch.tensor(hex_to_rgb(color), dtype=torch.float32, device=device)
            for color in palette.values()
        ])

        closest_colors = rgb_palette[closest_indices]
        result = closest_colors.reshape(h, w, 3).cpu().numpy().astype(np.uint8)

        # Preserve alpha channel if needed
        if preserve_alpha and img_array.shape[2] == 4:
            result = np.dstack([result, img_array[:, :, 3]])

        return result

    except ImportError:
        print("PyTorch not available. Install with: pip install torch")
        return None


def rgb_to_lab_gpu(rgb_tensor):
    """GPU-accelerated RGB to LAB conversion using PyTorch."""
    import torch

    # normalize
    rgb_norm = rgb_tensor / 255.0

    # Gamma correction (vectorized)
    gamma_corrected = torch.where(
        rgb_norm > GAMMA_THRESHOLD,
        torch.pow((rgb_norm + GAMMA_OFFSET) / GAMMA_DIVISOR, GAMMA_EXPONENT),
        rgb_norm / LINEAR_DIVISOR
    )


    # RGB to XYZ transformation matrix
    device = rgb_tensor.device
    transform_matrix = torch.tensor(TRNS_MAT, device=device)

    # Apply transformation
    if gamma_corrected.dim() == 3:  # Image tensor (H, W, 3)
        h, w, c = gamma_corrected.shape
        xyz = torch.matmul(gamma_corrected.reshape(-1, 3), transform_matrix.T).reshape(h, w, 3)
    else:  # Single pixel or batch of pixels
        xyz = torch.matmul(gamma_corrected, transform_matrix.T)

    # Normalize by D65 illuminant
    d65_illuminant = torch.tensor([D65_X, D65_Y, D65_Z], device=device)
    xyz_norm = xyz / d65_illuminant

    # XYZ to LAB transformation (vectorized)
    def f_transform(t):
        return torch.where(
            t > LAB_THRESHOLD,
            torch.pow(t, LAB_CUBE_ROOT),           # Cube root for bright colors
            (LAB_SLOPE * t + LAB_OFFSET)           # Linear for dark colors
        )

    f_xyz = f_transform(xyz_norm)

    # Calculate LAB values
    L = L_SCALE * f_xyz[..., 1] - L_OFFSET    # Lightness (0-100)
    a = A_SCALE * (f_xyz[..., 0] - f_xyz[..., 1])  # Green-Red axis
    b = B_SCALE * (f_xyz[..., 1] - f_xyz[..., 2])  # Blue-Yellow axis

    return torch.stack([L, a, b], dim=-1)


def rgb_to_hsv_gpu(rgb_tensor):
    """GPU-accelerated RGB to HSV conversion using PyTorch."""
    import torch

    rgb_norm = rgb_tensor / 255.0

    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]

    max_val, max_idx = torch.max(rgb_norm, dim=-1)
    min_val, _ = torch.min(rgb_norm, dim=-1)

    delta = max_val - min_val

    # Value component
    v = max_val

    # Saturation component
    s = torch.where(max_val != 0, delta / max_val, torch.zeros_like(max_val))

    # Hue component
    h = torch.zeros_like(max_val)

    # Red is max
    mask_r = (max_idx == 0) & (delta != 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360

    # Green is max
    mask_g = (max_idx == 1) & (delta != 0)
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360

    # Blue is max
    mask_b = (max_idx == 2) & (delta != 0)
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

    # Normalize hue to 0-1
    h = h / 360.0

    return torch.stack([h, s, v], dim=-1)


# Method CuPy
def convert_with_cupy(img_array: np.ndarray, preserve_alpha: bool = True) -> np.ndarray:
    """Convert image using CuPy GPU acceleration."""
    try:
        import cupy as cp
        print("Using CuPy GPU acceleration")

        # Prepare palette colors
        palette_colors = cp.array([hex_to_rgb(color) for color in GRUVBOX_PALETTE.values()], dtype=cp.float32)

        # Convert image to CuPy array
        h, w = img_array.shape[:2]
        rgb_data = cp.array(img_array[:, :, :3], dtype=cp.float32)

        # Reshape for vectorized operations
        pixels = rgb_data.reshape(-1, 3)

        # Vectorized distance calculation
        # Broadcasting: pixels[:, None, :] - palette_colors[None, :, :]
        diff = pixels[:, None, :] - palette_colors[None, :, :]
        distances = cp.sqrt(cp.sum(diff ** 2, axis=2))

        # Find closest colors
        closest_indices = cp.argmin(distances, axis=1)
        closest_colors = palette_colors[closest_indices]

        # Reshape back and convert to numpy
        result = closest_colors.reshape(h, w, 3).get().astype(np.uint8)

        # Preserve alpha channel if needed
        if preserve_alpha and img_array.shape[2] == 4:
            result = np.dstack([result, img_array[:, :, 3]])

        return result

    except ImportError:
        print("CuPy not available. Install with: pip install cupy-cuda12x")
        return None


# Method Numba CUDA (for custom kernels)
def convert_with_numba(img_array: np.ndarray, preserve_alpha: bool = True) -> np.ndarray:
    """Convert image using Numba CUDA acceleration."""
    try:
        from numba import cuda
        import math
        print("Using Numba CUDA acceleration")

        # Prepare palette
        palette_rgb = np.array([hex_to_rgb(color) for color in GRUVBOX_PALETTE.values()], dtype=np.float32)

        @cuda.jit
        def find_closest_color_kernel(pixels, palette, result):
            """CUDA kernel to find closest color for each pixel."""
            idx = cuda.grid(1)
            if idx < pixels.shape[0]:
                min_dist = float('inf')
                closest_idx = 0

                # Check against each palette color
                for p_idx in range(palette.shape[0]):
                    dist = 0.0
                    for c in range(3):  # RGB channels
                        diff = pixels[idx, c] - palette[p_idx, c]
                        dist += diff * diff
                    dist = math.sqrt(dist)

                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = p_idx

                # Copy closest color to result
                for c in range(3):
                    result[idx, c] = palette[closest_idx, c]

        h, w = img_array.shape[:2]
        pixels = img_array[:, :, :3].reshape(-1, 3).astype(np.float32)

        # Allocate GPU memory
        d_pixels = cuda.to_device(pixels)
        d_palette = cuda.to_device(palette_rgb)
        d_result = cuda.device_array((pixels.shape[0], 3), dtype=np.float32)

        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (pixels.shape[0] + threads_per_block - 1) // threads_per_block
        find_closest_color_kernel[blocks_per_grid, threads_per_block](d_pixels, d_palette, d_result)

        # Copy result back
        result = d_result.copy_to_host().reshape(h, w, 3).astype(np.uint8)

        # Preserve alpha channel if needed
        if preserve_alpha and img_array.shape[2] == 4:
            result = np.dstack([result, img_array[:, :, 3]])

        return result

    except ImportError:
        print("Numba not available. Install with: pip install numba")
        return None


# Fallback: Optimized NumPy version
def convert_with_numpy_optimized(img_array: np.ndarray, preserve_alpha: bool = True) -> np.ndarray:
    """Optimized NumPy version (CPU but vectorized)."""
    print("Using optimized NumPy (CPU)")

    # Prepare palette colors
    palette_colors = np.array([hex_to_rgb(color) for color in GRUVBOX_PALETTE.values()], dtype=np.float32)

    h, w = img_array.shape[:2]
    rgb_data = img_array[:, :, :3].astype(np.float32)

    # Reshape for vectorized operations
    pixels = rgb_data.reshape(-1, 3)

    # Vectorized distance calculation using broadcasting
    # pixels shape: (h*w, 3), palette_colors shape: (num_colors, 3)
    diff = pixels[:, np.newaxis, :] - palette_colors[np.newaxis, :, :]  # (h*w, num_colors, 3)
    distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (h*w, num_colors)

    # Find closest colors
    closest_indices = np.argmin(distances, axis=1)
    closest_colors = palette_colors[closest_indices]

    # Reshape back to image format
    result = closest_colors.reshape(h, w, 3).astype(np.uint8)

    # Preserve alpha channel if needed
    if preserve_alpha and img_array.shape[2] == 4:
        result = np.dstack([result, img_array[:, :, 3]])

    return result


def convert_image_to_gruvbox_gpu(input_path: str, output_path: str,
                                method: str = 'auto', preserve_alpha: bool = True,
                                color_method: str = 'lab', use_extended_palette: bool = True) -> None:
    """
    Convert an image to Gruvbox palette using GPU acceleration.

    Args:
        input_path: Path to input image
        output_path: Path to save converted image
        method: 'pytorch', 'cupy', 'numba', 'numpy', or 'auto'
        preserve_alpha: Whether to preserve alpha channel
        color_method: 'lab', 'hsv', 'weighted_rgb', 'gray_biased', or 'standard'
        use_extended_palette: Whether to use extended palette with more colors
    """
    print(f"Loading image: {input_path}")

    # Load image
    img = Image.open(input_path).convert('RGBA')
    width, height = img.size
    print(f"Image size: {width}x{height} ({width*height:,} pixels)")

    img_array = np.array(img)

    start_time = time.time()

    # Choose conversion method
    other_methods = {
        'numba': convert_with_numba,
        'numpy': convert_with_numpy_optimized
    }

    if method == 'auto':
        # Try methods in order of preference
        for method_name in ['pytorch', 'cupy', 'numba', 'numpy']:
            if method_name == 'pytorch':
                result = convert_with_pytorch(img_array, color_method, use_extended_palette, preserve_alpha)
            elif method_name == 'cupy':
                result = convert_with_cupy_advanced(img_array, color_method, use_extended_palette, preserve_alpha)
            else:
                result = other_methods[method_name](img_array, preserve_alpha)
            if result is not None:
                break
    else:
        if method not in ['pytorch', 'cupy'] + list(other_methods.keys()):
            raise ValueError(f"Unknown method: {method}. Choose from pytorch, cupy, {', '.join(other_methods.keys())}, or 'auto'")

        if method == 'pytorch':
            result = convert_with_pytorch(img_array, color_method, use_extended_palette, preserve_alpha)
        elif method == 'cupy':
            result = convert_with_cupy_advanced(img_array, color_method, use_extended_palette, preserve_alpha)
        else:
            result = other_methods[method](img_array, preserve_alpha)

        if result is None:
            print(f"Method {method} failed, falling back to NumPy")
            result = convert_with_numpy_optimized(img_array, preserve_alpha)

    conversion_time = time.time() - start_time
    print(f"Conversion completed in {conversion_time:.2f} seconds")
    print(f"Performance: {(width*height/conversion_time/1000000):.1f} megapixels/second")

    # Save result
    converted_img = Image.fromarray(result, 'RGBA' if result.shape[2] == 4 else 'RGB')

    if output_path.lower().endswith(('.jpg', '.jpeg')) and result.shape[2] == 4:
        # Convert to RGB for JPEG
        background = Image.new('RGB', converted_img.size, (255, 255, 255))
        background.paste(converted_img, mask=converted_img.split()[-1])
        converted_img = background

    converted_img.save(output_path, quality=95, optimize=True)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated Gruvbox image converter with advanced color matching",
        epilog="""
Examples:
  # Best quality (LAB color space + extended palette)
  python gruvbox_gpu_converter.py image.jpg output.jpg --color-method lab --extended-palette

  # Reduce gray over-selection
  python gruvbox_gpu_converter.py image.jpg output.jpg --color-method gray_biased

  # Preserve hue better
  python gruvbox_gpu_converter.py image.jpg output.jpg --color-method hsv

  # Force specific GPU method
  python gruvbox_gpu_converter.py image.jpg output.jpg --method pytorch --color-method lab
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('output', help='Output image file path')
    parser.add_argument('--method', choices=['auto', 'pytorch', 'cupy', 'numba', 'numpy'],
                       default='auto', help='GPU acceleration method (default: auto)')
    parser.add_argument('--color-method', choices=['lab', 'hsv', 'weighted_rgb', 'gray_biased', 'standard'],
                       default='lab', help='Color matching algorithm (default: lab - most perceptual)')
    parser.add_argument('--extended-palette', action='store_true', default=True,
                       help='Use extended palette with more colors (default: True)')
    parser.add_argument('--original-palette', action='store_true',
                       help='Use original Gruvbox palette only (overrides --extended-palette)')
    parser.add_argument('--no-alpha', action='store_true',
                       help='Do not preserve alpha channel')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Handle palette selection
    use_extended = args.extended_palette and not args.original_palette

    try:
        convert_image_to_gruvbox_gpu(
            args.input,
            args.output,
            method=args.method,
            preserve_alpha=not args.no_alpha,
            color_method=args.color_method,
            use_extended_palette=use_extended
        )

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

