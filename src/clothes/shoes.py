from clothes.character import CharacterBuilder
import os
from random import randint
from PIL import Image

class Shoes:
    """
    Class to add shoes
    """
    def __init__(self, base_path, width=800, height=800):
        self.base_path = base_path
        self.width = width
        self.height = height

        self.character_builder = CharacterBuilder(base_path)
        self.character_image = self.character_builder.build_character()

        self.positions = {
            'shoe': (480,630),
        }
        self.mirrored_positions = {
            'shoe': (280,630)
        }

        self.folder_images_shoe_color = self.get_random_shoe_folder()
        self.color_folder = [f for f in os.listdir(self.folder_images_shoe_color) if f.endswith('.png')]

        self.shoes = [shoe for shoe in self.color_folder]
   
        self.shoe = self.shoes[randint(0, len(self.shoes) - 1)] 

    def get_random_shoe_folder(self):
        shoes_dir = os.listdir(os.path.join(self.base_path, 'Shoes'))
        colors = [color for color in shoes_dir if color != '.DS_Store'] 
        
        random_color = colors[randint(0, len(colors) - 1)] 
        return os.path.join(self.base_path, 'Shoes', random_color)
        
    def add_to_character(self, character_image):

        shoe_path = os.path.join(self.folder_images_shoe_color, self.shoe)
        shoe_img = Image.open(shoe_path).convert('RGBA')
        character_image.paste(shoe_img, self.positions['shoe'], shoe_img)
        
        mirrored_shoe_img = shoe_img.transpose(Image.FLIP_LEFT_RIGHT)
        character_image.paste(mirrored_shoe_img, self.mirrored_positions['shoe'], mirrored_shoe_img)
