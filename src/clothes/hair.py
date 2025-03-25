import os
from random import randint
from PIL import Image

class Hair:
    """
    Class to add hair
    """
    def __init__(self, base_path, width=800, height=800):
        self.base_path = base_path
        self.width = width
        self.height = height

        self.positions = {
            'hair': (345,50),
            'man2_5': (345, 30)
        }

        self.folder_images_hair_color = self.get_random_hair_folder()
        self.color_folder = [f for f in os.listdir(self.folder_images_hair_color) if f.endswith('.png')]

        self.hairs = [hair for hair in self.color_folder]
        self.hair = self.hairs[randint(0, len(self.hairs) - 1)] 

    def get_random_hair_folder(self):
        hair_dir = os.listdir(os.path.join(self.base_path, 'Hair'))
        colors = [color for color in hair_dir if color != '.DS_Store'] 
        
        random_color = colors[randint(0, len(colors) - 1)] 
        return os.path.join(self.base_path, 'Hair', random_color)
        
    def add_to_character(self, character_image):

        hair_path = os.path.join(self.folder_images_hair_color, self.hair)
        hair_img = Image.open(hair_path).convert('RGBA')
        if "Man2" in hair_path or "Man3" in hair_path or "Man4" in hair_path  or "Man5" in hair_path:
            character_image.paste(hair_img, self.positions['man2_5'], hair_img)
        else:
            character_image.paste(hair_img, self.positions['hair'], hair_img)
