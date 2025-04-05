import os
import random
from PIL import Image

# Base directory for skin
base_skin_dir = "PNG/Skin"

# Create a new blank image (white background) with adjusted size
background_width = 96
background_height = 120
background = Image.new('RGBA', (background_width, background_height), "white")

# Function to load, resize, and place a part
def place_part(img, part_path, position_ratio, scale, mirror=False):
    part = Image.open(part_path)
    if mirror:
        part = part.transpose(Image.FLIP_LEFT_RIGHT)  # Mirror the image if needed
    new_width = int(part.width * scale)
    new_height = int(part.height * scale)
    part = part.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # Calculate position as a ratio of the canvas size
    new_position = (int(background_width * position_ratio[0] - new_width // 2),
                    int(background_height * position_ratio[1] - new_height // 2))
    img.paste(part, new_position, part)

# Function to generate an image with random skin
def generate_image():
    tint = random.choice([f for f in os.listdir(base_skin_dir) if os.path.isdir(os.path.join(base_skin_dir, f))])
    skin_dir = os.path.join(base_skin_dir, tint)

    scale_factor = 0.18  # Reduced scale factor for better fitting of clothing

    # Positions as ratios of the background dimensions
    positions = {
        'head': (0.5, 0.12),
        'neck': (0.5, 0.29),
        'arm': (0.79, 0.45),
        'hand': (0.95, 0.6),
        'leg': (0.68, 0.77),
        'arm_mirrored': (0.21, 0.45),
        'hand_mirrored': (0.05, 0.6),
        'leg_mirrored': (0.36, 0.77)
    }

    # Place body parts; original left parts
    parts = ['head', 'neck', 'arm', 'hand', 'leg']
    for part in parts:
        part_path = os.path.join(skin_dir, f"{tint.lower().replace(' ', '')}_{part}.png")
        place_part(background, part_path, positions[part], scale_factor)

    # Place mirrored parts; only right parts are mirrored
    mirrored_parts = ['arm', 'hand', 'leg']
    for part in mirrored_parts:
        part_path = os.path.join(skin_dir, f"{tint.lower().replace(' ', '')}_{part}.png")
        mirrored_position = positions[f"{part}_mirrored"]
        place_part(background, part_path, mirrored_position, scale_factor, mirror=True)

    # Save the image
    output_path = f"data/generated_image_{tint}.png"
    background.save(output_path)

# Generate a set number of images
number_of_images = 10
for _ in range(number_of_images):
    generate_image()
    background = Image.new('RGBA', (background_width, background_height))
