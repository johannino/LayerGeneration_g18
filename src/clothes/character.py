from PIL import Image
import os
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
            'leg': (self.center_x + 40, 470)
        }
        self.mirrored_positions = {
            'arm': (self.center_x - 160 - self.sizes['arm'][0], 260),
            'hand': (self.center_x - 210 - self.sizes['hand'][0], 390),
            'leg': (self.center_x - 20 - self.sizes['leg'][0], 470)
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
        
        resize_res = (100,100)
        resized = self.blank_image.resize(resize_res)
        resized.save(filename)

        #self.blank_image.save(filename)

    def show_character(self):
        self.blank_image.show()