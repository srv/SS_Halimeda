import os
from PIL import Image

path="/home/tintin/SS_Halimeda/data/1024_1024"

new_width=1024
new_height=1024

for img_name in os.listdir(path):
    path_img = os.path.join(path,img_name)
    image = Image.open(path_img)
    image = image.resize((new_width,new_height))
    image.save(path_img) 
