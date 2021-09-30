import tensorflow as tf
import utils

if __name__ == '__main__'
content_path = '/home/mzhang/Style_Transfer/content.jpg'
style_path = '/home/mzhang/Style_Transfer/style.jpg'

c_img = utils.load_image(content_path)
s_img = utils.load_image(style_path)

utils.show_images_with_objects([c_img, s_img], titles=['c', 's'])