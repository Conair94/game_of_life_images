#!/usr/bin/env python3

"""
Converts an image into an animated Conway's Game of Life simulation.

This script takes an input image, converts it to grayscale, and then generates
an initial Game of Life pattern based on the image's brightness. It then
simulates the evolution of this pattern for a specified number of frames and
saves the result as an animated GIF with a boomerang effect.

Requires:
- Pillow: `pip install Pillow`
- NumPy: `pip install numpy`
"""

import argparse
import os
from PIL import Image
import numpy as np

# A collection of 5x5 patterns, including oscillators and other dynamic forms,
# to create an interesting animation from a static image.
PATTERNS = [
    # Level 0: Empty
    np.zeros((5, 5), dtype=np.uint8),
    # Level 1: Block (Still Life)
    np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8),
    # Level 2: Blinker (Oscillator, period 2)
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8),
    # Level 3: Toad (Oscillator, period 2)
    np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8),
    # Level 4: Glider (Spaceship)
    np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.uint8),
    # Level 5: Pulsar (Period 3) - just the core
    np.array([
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0]
    ], dtype=np.uint8),
]

def compute_next_generation(grid):
    """
    Computes the next state of the Game of Life grid.
    
    Args:
        grid (np.ndarray): The current state of the grid (2D array of 0s and 1s).

    Returns:
        np.ndarray: The next state of the grid.
    """
    # Count live neighbors for each cell
    neighbors = (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) +
        np.roll(np.roll(grid, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(grid, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(grid, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    )

    # Apply Game of Life rules
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    birth = (grid == 0) & (neighbors == 3)
    
    new_grid = np.zeros_like(grid)
    new_grid[survive | birth] = 1
    
    return new_grid

def create_gol_animation(input_image_path, output_image_path, width_blocks, frames, duration):
    """
    Generates an animated GIF of a Game of Life simulation from an image.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the output GIF.
        width_blocks (int): The width of the output in pattern blocks.
        frames (int): The number of frames (generations) to simulate.
        duration (int): The duration of each frame in milliseconds.
    """
    try:
        image = Image.open(input_image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_image_path}")
        return

    aspect_ratio = image.height / image.width
    height_blocks = int(width_blocks * aspect_ratio)

    resized_image = image.resize((width_blocks, height_blocks))
    image_array = np.array(resized_image)

    block_height, block_width = PATTERNS[0].shape
    output_height_pixels = height_blocks * block_height
    output_width_pixels = width_blocks * block_width
    
    grid = np.zeros((output_height_pixels, output_width_pixels), dtype=np.uint8)

    num_patterns = len(PATTERNS)
    grayscale_step = 256 / num_patterns

    for r in range(height_blocks):
        for c in range(width_blocks):
            grayscale_value = image_array[r, c]
            pattern_index = int(grayscale_value / grayscale_step)
            pattern = PATTERNS[pattern_index]
            
            r_offset = r * block_height
            c_offset = c * block_width
            grid[r_offset:r_offset + block_height, c_offset:c_offset + block_width] = pattern

    image_frames = []
    for _ in range(frames):
        frame_image = Image.fromarray(grid * 255)
        image_frames.append(frame_image)
        grid = compute_next_generation(grid)

    # Create boomerang effect
    boomerang_frames = image_frames + image_frames[-2:0:-1]

    boomerang_frames[0].save(
        output_image_path,
        save_all=True,
        append_images=boomerang_frames[1:],
        duration=duration,
        loop=0  # Loop forever
    )
    print(f"Successfully created Game of Life animation at {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert an image to an animated Conway's Game of Life GIF.")
    parser.add_argument(
        'input_image', 
        help='Path to the input image file.'
    )
    parser.add_argument(
        '-o', '--output', 
        dest='output_image',
        help='Path to save the output GIF. Defaults to adding a `_gol_anim` suffix to the input file name.'
    )
    parser.add_argument(
        '-w', '--width', 
        dest='width', 
        type=int, 
        default=200,
        help='Width of the output in pattern blocks (default: 100).'
    )
    parser.add_argument(
        '-f', '--frames',
        dest='frames',
        type=int,
        default=50,
        help='Number of frames (generations) to simulate (default: 100).'
    )
    parser.add_argument(
        '-d', '--duration',
        dest='duration',
        type=int,
        default=100,
        help='Duration of each frame in milliseconds (default: 100).'
    )

    args = parser.parse_args()

    if not args.output_image:
        file_name, _ = os.path.splitext(args.input_image)
        args.output_image = f"{file_name}_gol_anim.gif"

    create_gol_animation(args.input_image, args.output_image, args.width, args.frames, args.duration)

if __name__ == '__main__':
    main()
