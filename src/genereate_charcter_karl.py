from clothes.character import CharacterBuilder
from clothes.shirt_arm import Shirts
from clothes.shoes import Shoes
from clothes.pants import Pants
from clothes.hair import Hair
from clothes.face import CompleteFace
import os

def create_and_save_character(character, item_class, base_path, data_folder, item_name, index):
    item = item_class(base_path)
    character.add_item(item)
    character.save_character(os.path.join(data_folder, item_name, f"{item_name}_character_{index}.png"))

if __name__ == "__main__":
    base_path = "../PNG"
    data_folder = "../data"

    for i in range(10):
        character = CharacterBuilder(base_path)
        character.build_character()
        character.save_character(os.path.join(data_folder, "base", f"base_character_{i}.png"))

        items = [
            (Shirts, "shirt"),
            (Shoes, "shoe"),
            (Pants, "pants"),
            (Hair, "hair"),
            (CompleteFace, "face")
        ]

        for item_class, item_name in items:
            create_and_save_character(character, item_class, base_path, data_folder, item_name, i)