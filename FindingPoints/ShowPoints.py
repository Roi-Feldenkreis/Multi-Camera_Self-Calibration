import os
import glob
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load configuration (assuming it is a JSON file for simplicity)
def read_configuration():
    with open('config.json', 'r') as f:
        return json.load(f)

# Load image data
def load_data(config):
    data = {}
    for cam_id in config['files']['idxcams']:
        if cam_id > -1:
            img_pattern = os.path.join(config['paths']['img'], f"cam{cam_id}", f"*{config['files']['imgext']}")
        else:
            img_pattern = os.path.join(config['paths']['img'], f"*{config['files']['imgext']}")
        data[cam_id] = sorted(glob.glob(img_pattern))
    return data

# Draw points on images and save
def draw_points_on_images(config, data, loaded, save_img=True, compose_imgs=True):
    font = ImageFont.load_default()

    if not compose_imgs:
        for cam_id, images in data.items():
            for idx, image_path in enumerate(images):
                img = Image.open(image_path)
                draw = ImageDraw.Draw(img)

                # Annotation text
                text = f"Camera: {cam_id:02d} Frame: {idx:03d}"
                draw.text((15, 20), text, fill="green", font=font)

                if loaded['IdMat'][cam_id][idx] > 0:
                    x, y = loaded['Ws'][3*cam_id-2][idx], loaded['Ws'][3*cam_id-1][idx]
                    draw.ellipse((x-12, y-12, x+12, y+12), outline="green", width=2)
                    draw.ellipse((x-7, y-7, x+7, y+7), outline="red", width=1)
                else:
                    draw.text((15, 40), "No point found", fill="green", font=font)

                if save_img:
                    output_path = os.path.join(config['paths']['data'], f"point.cam{cam_id}.{idx}.jpg")
                    img.save(output_path)

    if compose_imgs:
        for idx in range(len(loaded['IdMat'][0])):
            composed_images = []
            for cam_id, images in data.items():
                img = Image.open(images[idx])
                draw = ImageDraw.Draw(img)

                # Annotation text
                text = f"Camera: {cam_id:02d} Frame: {idx:03d}"
                draw.text((15, 20), text, fill="green", font=font)

                if loaded['IdMat'][cam_id][idx] > 0:
                    x, y = loaded['Ws'][3*cam_id-2][idx], loaded['Ws'][3*cam_id-1][idx]
                    draw.ellipse((x-12, y-12, x+12, y+12), outline="green", width=2)
                    draw.ellipse((x-7, y-7, x+7, y+7), outline="red", width=1)
                else:
                    draw.text((15, 40), "No point found", fill="green", font=font)

                composed_images.append(img)

            # Create a montage of images
            widths, heights = zip(*(i.size for i in composed_images))
            total_width = sum(widths)
            max_height = max(heights)
            montage = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for img in composed_images:
                montage.paste(img, (x_offset, 0))
                x_offset += img.width

            if save_img:
                output_path = os.path.join(config['paths']['data'], f"points.composed.{idx:04d}.eps")
                montage.save(output_path)

# Main execution
if __name__ == "__main__":
    config = read_configuration()
    data = load_data(config)
    loaded = {
        "IdMat": np.loadtxt(config['paths']['idmat']),
        "Ws": np.loadtxt(config['paths']['ws'])
    }
    draw_points_on_images(config, data, loaded, save_img=True, compose_imgs=True)
