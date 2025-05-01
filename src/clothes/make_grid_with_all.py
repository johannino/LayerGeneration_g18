import matplotlib.pyplot as plt
from character import CharacterBuilder
from face import CompleteFace
from hair import Hair
from pants import Pants
from shirt_arm import Shirts
from shoes import Shoes
import os

def make_grid_with_all(base_path, data_folder, resolution=(256, 256)):
    """
    Create a grid of characters with different clothing items.
    """
    # Create a grid of characters
    character = CharacterBuilder(base_path)
    character.build_character()

    # Create a figure to hold the grid
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle("Character with Different Clothing Items", fontsize=16)

    # List of clothing items and their positions in the grid
    clothing_items = [
        (Shirts, "shirt", (0, 0)),
        (Pants, "pants", (0, 1)),
        (Shoes, "shoe", (0, 2)),
        (Hair, "hair", (0, 3)),
        (CompleteFace, "face", (0, 4))
    ]

    # Add each clothing item to the character and plot it
    for item_class, item_name, pos in clothing_items:
        item = item_class(base_path)
        character.add_item(item)
        
        # Save the character with the item
        save_path = os.path.join(data_folder, f"{item_name}_character.png")
        character.save_character(resolution, save_path)
        
        # Load and display the image
        img = plt.imread(save_path)
        axs[pos].imshow(img)
        axs[pos].axis('off')
        axs[pos].set_title(item_name.capitalize())

    plt.tight_layout()
    plt.show()
    plt.savefig("character_grid.png")
    print("Grid saved as character_grid.png")

if __name__ == "__main__":
    base_path = "../../PNG"
    data_folder = "../../data"

    shoe = Shoes(base_path)
    face = CompleteFace(base_path)
    hair = Hair(base_path)
    pants = Pants(base_path)
    shirt = Shirts(base_path)
    character = CharacterBuilder(base_path)
    character.build_character()

    #plt.imshow(character.blank_image)
    #plt.savefig("character.png")
    #exit()


    fig, axs = plt.subplots(1, 6, figsize=(15, 3))
    fig.suptitle("Different Clothing Items", fontsize=16)

    axs[5].imshow(plt.imread(os.path.join(face.face)))
    axs[5].axis('off')
    axs[5].set_title("Face")

    axs[4].imshow(plt.imread(os.path.join(hair.folder_images_hair_color, hair.hair)))
    axs[4].axis('off')
    axs[4].set_title("Hair")

    axs[3].imshow(plt.imread(os.path.join(pants.folder_images_pants_color, pants.choosen_pants)))
    axs[3].axis('off')
    axs[3].set_title("Pants")

    axs[1].imshow(plt.imread(os.path.join(shirt.folder_images_shirt_color, shirt.shirt)))
    axs[1].axis('off')
    axs[1].set_title("Shirt")

    axs[2].imshow(plt.imread(os.path.join(shoe.folder_images_shoe_color, shoe.shoe)))
    axs[2].axis('off')
    axs[2].set_title("Shoe")

    axs[0].imshow(character.blank_image)
    axs[0].axis('off')
    axs[0].set_title("Character")

    plt.tight_layout()
    plt.savefig("clothing_items.png")

    #shoe_img = os.path.join(shoe.folder_images_shoe_color, shoe.shoe)
    #img = plt.imread(shoe_img)
    #plt.imshow(img)
    #plt.savefig('shoe.png')

    #make_grid_with_all(base_path, data_folder)
    print("Grid created and saved successfully.")