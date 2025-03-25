from clothes.character import CharacterBuilder
import os
from random import randint
from PIL import Image

class Shirts:
    """
    Class to add shirts to the character.
    """
    def __init__(self, base_path, width=800, height=800):
        self.base_path = base_path
        self.width = width
        self.height = height

        self.character_builder = CharacterBuilder(base_path)

        self.positions = {
            'shirt': (self.character_builder.positions['neck'][0]-30, self.character_builder.positions['neck'][1]+15),
            'arm': self.character_builder.positions['arm'],
        }
        self.mirrored_positions = {
            'arm': self.character_builder.mirrored_positions['arm'],
            'short': (self.character_builder.positions['arm'][0]-258, self.character_builder.positions['arm'][1]),
            'shorter': (self.character_builder.positions['arm'][0]-205, self.character_builder.positions['arm'][1])
        }

        self.folder_images_shirt_color = self.get_random_shirt_folder()
        self.color_folder = [f for f in os.listdir(self.folder_images_shirt_color) if f.endswith('.png')]

        self.arms = [arm for arm in self.color_folder if 'Arm' in arm or 'arm' in arm]
        self.shirts = [shirt for shirt in self.color_folder if 'Shirt' in shirt or 'shirt' in shirt]

        self.arm = self.arms[randint(0, len(self.arms) - 1)]
        self.shirt = self.shirts[randint(0, len(self.shirts) - 1)]

    def get_random_shirt_folder(self):
        shirts_dir = os.listdir(os.path.join(self.base_path, 'Shirts'))
        colors = [color for color in shirts_dir if color != '.DS_Store'] 
        
        random_color = colors[randint(0, len(colors) - 1)] 
        return os.path.join(self.base_path, 'Shirts', random_color)
        
    def add_to_character(self, character_image):

        shirt_path = os.path.join(self.folder_images_shirt_color, self.shirt)
        shirt_img = Image.open(shirt_path).convert('RGBA')
        character_image.paste(shirt_img, self.positions['shirt'], shirt_img)
    
        arm_path = os.path.join(self.folder_images_shirt_color, self.arm)
        arm_img = Image.open(arm_path).convert('RGBA')
        character_image.paste(arm_img, self.positions['arm'], arm_img)
        
        mirrored_arm_img = arm_img.transpose(Image.FLIP_LEFT_RIGHT)
        if 'shorter' in arm_path:
            character_image.paste(mirrored_arm_img, self.mirrored_positions['shorter'], mirrored_arm_img)
        elif 'short' in arm_path:
            character_image.paste(mirrored_arm_img, self.mirrored_positions['short'], mirrored_arm_img)
        else:
            character_image.paste(mirrored_arm_img, self.mirrored_positions['arm'], mirrored_arm_img)
