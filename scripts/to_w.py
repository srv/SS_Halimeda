import numpy
import os
from PIL import Image

folder = "/home/plome/DATA/INVHALI/sets/semantic/backgrounds/masks"

for filename in os.listdir(folder):

    f = os.path.join(folder, filename)
    f_new=f.split(".")[0]+"_gt.png"
    im = numpy.asarray(Image.open(f))
    im2=numpy.zeros(im.shape)
    # im2[:,:] = 0

    im2 = Image.fromarray(numpy.uint8(im2))

    im2.save(f_new)




	

