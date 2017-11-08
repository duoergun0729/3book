#-*- coding:utf-8 –*-

from PIL import Image

file_name="pig.jpeg"

pig=Image.open(file_name)

print pig.mode

print pig.size

pig_L=pig.convert("L")

pig_L.save("pig-L.jpeg")

pig_1=pig.convert("1")

pig_1.save("pig-1.jpeg")