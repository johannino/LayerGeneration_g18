from clothes.character import CharacterBuilder
from clothes.shirt_arm import Shirts
from clothes.shoes import Shoes
from clothes.pants import Pants
from clothes.hair import Hair
from clothes.face import CompleteFace

from time import sleep

def genereate_random_charcter(imgs_path):
    items = ["shirt"]
    items_generated = [Shirts(imgs_path)]
    return items, items_generated

if __name__ == "__main__":
    base_path = "../PNG" 
    figures_folder = "../figures"

    for i in range(10):
        character = CharacterBuilder(base_path)

        character.build_character()

        character.save_character("base_character.png")

        shirt = Shirts(base_path)
        character.add_item(shirt)

        character.save_character("character_with_shirt.png")

        shoe = Shoes(base_path)
        character.add_item(shoe)

        pants = Pants(base_path)
        character.add_item(pants)

        hair = Hair(base_path)
        character.add_item(hair)

        face = CompleteFace(base_path)
        character.add_item(face)
        character.save_character("character_with_face.png")

        sleep(1.5)


