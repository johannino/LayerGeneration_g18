import os
from PIL import Image
from tqdm import tqdm
from random import randint

class CharacterBuilder:
    def __init__(self, base_path, width=800, height=800):
        self.base_path = base_path
        self.width = width
        self.height = height
        self.blank_image = Image.new('RGBA', (self.width, self.height), color=(255, 255, 255, 0))
        self.items = []  

        self.sizes = {
            'head': (120, 120),
            'neck': (50, 50),
            'arm': (50, 120),
            'hand': (40, 40),
            'leg': (60, 160)
        }
        self.center_x = self.width // 2
        self.positions = {
            'head': (self.center_x - self.sizes['head'][0] // 2, 50),
            'neck': (self.center_x - self.sizes['neck'][0] // 2 + 7, 235),
            'arm': (self.center_x + 100, 260),
            'hand': (self.center_x + 250, 390),
            'leg': (self.center_x + 60, 430)
        }
        self.mirrored_positions = {
            'arm': (self.center_x - 160 - self.sizes['arm'][0], 260),
            'hand': (self.center_x - 210 - self.sizes['hand'][0], 390),
            'leg': (self.center_x - 20 - self.sizes['leg'][0], 430)
        }
        self.folder_images_skin_tint_1 = self.get_random_skin_folder()
        self.tint_1_folder = [f for f in os.listdir(self.folder_images_skin_tint_1) if f.endswith('.png')]

    def get_random_skin_folder(self):
        different_skins = os.listdir(os.path.join(self.base_path, 'Skin'))
        nr_of_skins = len(different_skins) - 1
        skin_nr = randint(1, nr_of_skins)
        return os.path.join(self.base_path, f'Skin/Tint {skin_nr}')

    def build_character(self):
        for image_name in self.tint_1_folder:
            img_path = os.path.join(self.folder_images_skin_tint_1, image_name)
            img = Image.open(img_path).convert('RGBA')

            for part in self.positions:
                if part in image_name:
                    self.blank_image.paste(img, self.positions[part], img)

                    if part in self.mirrored_positions:
                        mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        self.blank_image.paste(mirrored_img, self.mirrored_positions[part], mirrored_img)

        self.apply_items()
        

    def add_item(self, item):
        self.items.append(item)

        if self.blank_image:
            item.add_to_character(self.blank_image)

    def apply_items(self):
        for item in self.items:
            item.add_to_character(self.blank_image)

    def save_character(self, filename="created_character.png"):
        self.blank_image.save(filename)
        print(f"Character saved as {filename}")

    def show_character(self):
        self.blank_image.show()


class Shirts:
    """
    Class to add shirts to the character.
    """
    def __init__(self, base_path, width=800, height=800):
        self.base_path = base_path
        self.width = width
        self.height = height

        self.character_builder = CharacterBuilder(base_path)
        self.character_image = self.character_builder.build_character()

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

        self.arm = self.arms[randint(0, len(self.arms) - 1)] if len(self.arms) > 0 else None
        self.shirt = self.shirts[randint(0, len(self.shirts) - 1)] if len(self.shirts) > 0 else None

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

    def save_character_with_shirt(self, filename="character_with_shirt.png"):
        self.character_image.save(filename)

    def show_character_with_shirt(self):
        self.character_image.show()


if __name__ == "__main__":
    base_path = "/Users/karlfindhansen/Downloads/kenney_modular-characters/PNG" 

    character = CharacterBuilder(base_path)

    character.build_character()

    character.save_character("base_character.png")
    character.show_character()

    shirt = Shirts(base_path)
    character.add_item(shirt)

    character.save_character("character_with_shirt.png")
    character.show_character()