import numpy as np
import skimage.color as skc
from skimage.draw import circle
import skimage.io as ski
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns
from PIL import Image


class ImageSaver:

    palette = None

    def save_image(self,flat_im, number, locations):
	locations_floated = locations.astype(np.int64)
        flat_im = np.squeeze(flat_im)
        image_rgb = skc.gray2rgb(flat_im)
        image_grey = skc.rgb2gray(flat_im)
        Image.fromarray(np.uint8(flat_im)).convert('LA').save("clean-images/output-%d.tiff" % number)
        ski.use_plugin('tifffile')
        filters, dims = locations.shape

        if self.palette is None:
            self.define_colors(filters)

        for i in range(0, filters):
            rr, cc = circle(locations_floated[i, 0], locations_floated[i, 1], 2, flat_im.shape)
            image_rgb[rr, cc, :] = tuple(i* 255 for i in self.palette[i])

        #plt.imsave(file_name, image_rgb)
        Image.fromarray(np.uint8(image_rgb)).save("output-images/output-%d.tiff" % number)

        return None

    def define_colors(self, filters):
        self.palette = sns.color_palette(palette='bright', n_colors=filters)

