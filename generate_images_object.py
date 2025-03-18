import os
import random
from PIL import Image

class CharacterPart:
    def __init__(self, image_path, scale, position_ratio, mirror=False):
        self.image_path = image_path
        self.scale = scale
        self.position_ratio = position_ratio
        self.mirror = mirror
        self.image = None

    def load_and_process(self, canvas_size):
        self.image = Image.open(self.image_path)
        if self.mirror:
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = self.image.size
        new_width = int(width * self.scale)
        new_height = int(height * self.scale)
        self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        x = int(canvas_size[0] * self.position_ratio[0] - new_width // 2)
        y = int(canvas_size[1] * self.position_ratio[1] - new_height // 2)
        return self.image, (x, y)

class Character:
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size
        self.background = Image.new('RGBA', self.canvas_size, "white")
        self.parts = []

    def add_part(self, part):
        self.parts.append(part)

    def assemble_character(self):
        for part in self.parts:
            image, position = part.load_and_process(self.canvas_size)
            self.background.paste(image, position, image)

    def save(self, path):
        self.background.save(path)

class CharacterGenerator:
    def __init__(self, skin_dir, shirt_dir, output_dir, canvas_size=(96, 120)):
        self.skin_dir = skin_dir
        self.shirt_dir = shirt_dir
        self.output_dir = output_dir
        self.canvas_size = canvas_size
        self.scale_factor = 0.18
        self.shirt_scale_factor = 0.25  # Slightly larger for shirts

    def get_random_shirt(self):
        """Selects a random shirt color and random components (shirt + sleeves)."""
        color = random.choice(os.listdir(self.shirt_dir))  # Choose random shirt color
        shirt_path = os.path.join(self.shirt_dir, color)
        shirts = [f for f in os.listdir(shirt_path) if "shirt" in f.lower() and f.endswith(('.png', '.jpg'))]
        sleeves = [f for f in os.listdir(shirt_path) if "arm" in f.lower() and f.endswith(('.png', '.jpg'))]


        selected_shirt = random.choice(shirts)
        selected_sleeve = random.choice(sleeves)

        return {
            "color": color,
            "shirt": os.path.join(shirt_path, selected_shirt),
            "sleeve": os.path.join(shirt_path, selected_sleeve)
        }

    def generate(self, number_of_images):
        for _ in range(number_of_images):
            tint = random.choice([f for f in os.listdir(self.skin_dir) if os.path.isdir(os.path.join(self.skin_dir, f))])
            skin_dir = os.path.join(self.skin_dir, tint)
            character = Character(self.canvas_size)

            # Body part positions
            positions = {
                'head': (0.5, 0.12), 'neck': (0.5, 0.29),
                'arm': (0.79, 0.45), 'hand': (0.95, 0.6), 'leg': (0.68, 0.77),
                'arm_mirrored': (0.21, 0.45), 'hand_mirrored': (0.05, 0.6), 'leg_mirrored': (0.36, 0.77)
            }

            # Add skin layer (body parts)
            parts = ['head', 'neck', 'arm', 'hand', 'leg']
            for part in parts:
                part_path = os.path.join(skin_dir, f"{tint.lower()}_{part}.png")
                character.add_part(CharacterPart(part_path, self.scale_factor, positions[part], False))
                if part in ['arm', 'hand', 'leg']:  # Add mirrored versions
                    mirrored_part_path = part_path
                    character.add_part(CharacterPart(mirrored_part_path, self.scale_factor, positions[f"{part}_mirrored"], True))

            # Add clothing layer (shirt + sleeves)
            shirt_info = self.get_random_shirt()
            shirt_position = (0.5, 0.50)  # Center shirt
            character.add_part(CharacterPart(shirt_info["shirt"], self.shirt_scale_factor, shirt_position, False))

            ### Dynamically adjust sleeves to the **upper part of the arms**
            arm_top_y = positions['arm'][1] - 0.05  # Adjust upwards slightly
            sleeve_offset_x = 0.06  # Move slightly toward the torso

            # Correct sleeve positions dynamically based on arms
            sleeve_left_position = (positions['arm'][0] - sleeve_offset_x, arm_top_y)
            sleeve_right_position = (positions['arm_mirrored'][0] + sleeve_offset_x, arm_top_y)

            # Add sleeves (aligned with top of the arms)
            character.add_part(CharacterPart(shirt_info["sleeve"], self.scale_factor, sleeve_left_position, False))
            character.add_part(CharacterPart(shirt_info["sleeve"], self.scale_factor, sleeve_right_position, True))

            # Assemble and save the character
            character.assemble_character()
            output_path = os.path.join(self.output_dir, f"generated_image_{tint}_{shirt_info['color']}.png")
            character.save(output_path)
            print(f"Generated {output_path}")


# Usage
generator = CharacterGenerator(
    skin_dir="/Users/johanverrecchia/Downloads/kenney_modular-characters/PNG/Skin",
    shirt_dir="/Users/johanverrecchia/Downloads/kenney_modular-characters/PNG/Shirts",
    output_dir="/Users/johanverrecchia/Downloads/kenney_modular-characters/data"
)
generator.generate(10)
