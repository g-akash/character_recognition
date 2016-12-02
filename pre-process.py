from skimage import io, filters
# import Image
from skimage import transform
import numpy
from scipy import misc
import time

train_size = 17205
test_size = 1829

#start_time = datetime.now()

for i in range(train_size):
	buf = './train/{}.png'.format(i)
	buf1 = './train_small/{}.png'.format(i)
	image = io.imread(buf)
	image = numpy.invert(image)
	image = filters.gaussian(image, 5)
	image = transform.resize(image, (80, 80))
	misc.imsave(buf1, image)
	# io.imsave(buf1, image)
	# image = Image.open(buf)
	# image = image.resize((80, 80), Image.ANTIALIAS)
	# image.save(buf1, "PNG")

for i in range(test_size):
	buf = './valid/{}.png'.format(i)
	buf1 = './valid_small/{}.png'.format(i)
	image = io.imread(buf)
	image = numpy.invert(image)
	image = filters.gaussian(image, 5)
	image = transform.resize(image, (80, 80))
	misc.imsave(buf1, image)
	# image = Image.open(buf)
	# image = image.resize((80, 80), Image.ANTIALIAS)
	# image.save(buf1, "PNG")


#print("Time taken:", datetime.now() - start_time);
