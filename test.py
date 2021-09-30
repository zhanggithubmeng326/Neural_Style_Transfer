import matplotlib.pyplot as plt
import tensorflow as tf
import utils


content_path = '/home/mzhang/Style_Transfer/content.jpg'
style_path = '/home/mzhang/Style_Transfer/style.jpg'

c_img = utils.load_image(content_path)
s_img = utils.load_image(style_path)

if __name__ == '__main__':
    plt.subplot(1, 2, 1)
    utils.imshow(c_img, 'content_img')
    
    plt.subplot(1, 2, 2)
    utils.imshow(s_img, 'style_img')