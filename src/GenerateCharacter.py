from clothes.character import CharacterBuilder
from clothes.shirt_arm import Shirts
from clothes.shoes import Shoes
from clothes.pants import Pants
from clothes.hair import Hair
from clothes.face import CompleteFace
import os
from tqdm import tqdm

def create_and_save_character(character, item_class, base_path, data_folder, item_name, index, resolution):
    item = item_class(base_path)
    character.add_item(item)
    
    save_path = os.path.join(data_folder, item_name, f"{item_name}_character_{index}.png")
    character.save_character(resolution, save_path)

if __name__ == "__main__":
    base_path = "../PNG"
    data_folder = "../data"

    for i in tqdm(range(250), desc='Generating layers in characters'):
        character = CharacterBuilder(base_path)
        character.build_character()

        resolution = (100,100)
        save_path_base = os.path.join(data_folder, "base", f"base_character_{i}.png")
        character.save_character(resolution, save_path_base)

        items = [
            (Shirts, "shirt"),
            (Shoes, "shoe"),
            (Pants, "pants"),
            (Hair, "hair"),
            (CompleteFace, "face")
        ]

        for item_class, item_name in items:
            create_and_save_character(character, item_class, base_path, data_folder, item_name, i, resolution)