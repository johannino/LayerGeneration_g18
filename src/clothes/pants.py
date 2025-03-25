from clothes.character import CharacterBuilder
import os
from random import randint
from PIL import Image

class Pants:
    """
    Class to add pants to the character.
    """
    def __init__(self, base_path, width=800, height=800):
        self.base_path = base_path
        self.width = width
        self.height = height

        self.character_builder = CharacterBuilder(base_path)
        self.character_image = self.character_builder.build_character()

        self.positions = {
            'pants': (self.character_builder.positions['leg'][0]-15, self.character_builder.positions['leg'][1]),
            'undies': (self.character_builder.positions['leg'][0]-90, self.character_builder.positions['leg'][1]-45)
        }
        self.mirrored_positions = {
            'pants': self.character_builder.mirrored_positions['leg'],
        }

        self.folder_images_pants_color = self.get_random_pants_folder()
        self.color_folder = [f for f in os.listdir(self.folder_images_pants_color) if f.endswith('.png')]

        self.pants = [pants for pants in self.color_folder if '_' in pants]
        self.undies = [pants for pants in self.color_folder if '_' not in pants]

        self.choosen_pants = self.pants[randint(0, len(self.pants) - 1)]
        self.undie = self.undies[randint(0, len(self.undies) - 1)]

    def get_random_pants_folder(self):
        pants_dir = os.listdir(os.path.join(self.base_path, 'Pants'))
        colors = [color for color in pants_dir if color != '.DS_Store'] 
        
        random_color = colors[randint(0, len(colors) - 1)] 
        return os.path.join(self.base_path, 'Pants', random_color)
        
    def add_to_character(self, character_image):

        pants_path = os.path.join(self.folder_images_pants_color, self.choosen_pants)
        pants_img = Image.open(pants_path).convert('RGBA')
        character_image.paste(pants_img, self.positions['pants'], pants_img)

        mirrored_pants_img = pants_img.transpose(Image.FLIP_LEFT_RIGHT)
        character_image.paste(mirrored_pants_img, self.mirrored_positions['pants'], mirrored_pants_img)

        undies_path = os.path.join(self.folder_images_pants_color, self.undie)
        undies_img = Image.open(undies_path).convert('RGBA')
        character_image.paste(undies_img, self.positions['undies'], undies_img)
        
        