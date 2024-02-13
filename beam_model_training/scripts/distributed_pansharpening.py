#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import concurrent.futures

def pansharpen_image(ms_path, pan_directory, output_directory):
    """
    Pan-sharpens a given multispectral image using its corresponding panchromatic image.

    Parameters:
    ms_path (Path): A Path object pointing to the multispectral image file.
    pan_directory (Path): A Path object representing the directory containing panchromatic images.
    output_directory (Path): A Path object representing the directory where the pan-sharpened images will be saved.

    Returns:
    str: A message indicating the status of the pan-sharpening process for the image. 
    """

    # Construct the corresponding panchromatic image filename
    pan_image_name = ms_path.name.replace('M3DS', 'P3DS')
    
    # Full paths for the input and output images
    pan_image_paths = list(pan_directory.rglob(pan_image_name))
    if len(pan_image_paths) > 0:
        pan_image_path = pan_image_paths[0]
        output_image_path = output_directory / ms_path.name.replace('M3DS', 'PS3DS')
    
        if not output_image_path.exists():
            # Construct the GDAL pansharpen command
            command = [
                'gdal_pansharpen.py',
                str(pan_image_path),
                str(ms_path),
                str(output_image_path),
                '-r', 'nearest',  # Resampling method, adjust as needed
                '-co', 'COMPRESS=DEFLATE'  
            ]
            
            # Execute the command
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                return f"Error processing {ms_path.name}: {e}"
        else:
            return f"Pan-sharpened image already exists for {ms_path.name}, skipping."
    else:
        return f"No panchromatic match found for {ms_path.name}, skipping."


def main(ms_directory, pan_directory, output_directory):
    # List all multispectral images

    if not ms_directory.exists() or not pan_directory.exists() or not any(pan_directory.rglob('*')):
        raise FileNotFoundError(
            f"The {'multispectral' if not ms_directory.exists() else 'panchromatic'} directory \
                is empty or does not exist.")

    # Check if output directory exists, create it if it does not
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    
    ms_images = []
    ms_images.extend(ms_directory.rglob('*.tif'))
    ms_images.extend(ms_directory.rglob('*.tiff'))
    ms_images.extend(ms_directory.rglob('*.TIF'))
    ms_images.extend(ms_directory.rglob('*.TIFF'))

    # Remove duplicates if any (in case the filesystem is case-insensitive)
    ms_images = list(set(ms_images))

    if not ms_images:
        raise ValueError("No multispectral images found in the specified directory.")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(
            pansharpen_image, 
            ms_images, 
            [pan_directory]*len(ms_images), 
            [output_directory]*len(ms_images)
            ))

    for result in results:
        if result is not None:
            print(result)

    print("Pan-sharpening process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pan-sharpening script.')
    parser.add_argument('mul_dir', type=str, help='Directory containing multispectral images')
    parser.add_argument('pan_dir', type=str, help='Directory containing panchromatic images')
    parser.add_argument('output_dir', type=str, help='Output directory for pan-sharpened images')

    args = parser.parse_args()

    mul_dir = Path(args.mul_dir)
    pan_dir = Path(args.pan_dir)
    output_dir = Path(args.output_dir)

    main(mul_dir, pan_dir, output_dir)
