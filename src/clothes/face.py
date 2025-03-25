import os
from random import randint
from PIL import Image

class CompleteFace:
    """
    Class to add shirts to the character.
    """
    def __init__(self, base_path, width=800, height=800):
        self.base_path = base_path
        self.width = width
        self.height = height

        self.positions = {
            'face': (370,100)
        }

        self.face = self.get_random_face()

    def get_random_face(self):
        face_dir = os.listdir(os.path.join(self.base_path,  'Face', 'Completes'))
        faces = [face for face in face_dir if face != '.DS_Store'] 
        
        random_face = faces[randint(0, len(faces) - 1)] 
        return os.path.join(self.base_path, 'Face', 'Completes', random_face)
        
    def add_to_character(self, character_image):

        face_path = self.face
        face_img = Image.open(face_path).convert('RGBA')
        character_image.paste(face_img, self.positions['face'], face_img)
    