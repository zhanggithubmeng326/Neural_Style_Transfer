import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# converts a tensor to an image
def tensor_to_image(tensor):
    tensor_shape = tf.shape(tensor)
    num_element = tf.shape(tensor_shape)
    if num_element > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)


def tensor_to_img(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# loads an image as a tensor and scales it to 512 pixels
def load_image(img_path):
    max_dim = 512
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]    # [height, width]
    shape = tf.cast(shape, tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    #image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


def load_images(content_path, style_path):
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    return content_image, style_image


# displays an image with a title
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)


# displays a row of images with titles
def show_images_with_objects(images, titles):
    if len(images) != len(titles):
        return
    plt.figure(figsize=(20,12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)


# clips the image pixel values by giving min and max
def clip_image_values(image, min_value=0.0, max_value=255.0):
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def preprocess_image(image):
    image = tf.cast(image, dtype=tf.float32)
    image = (image / 127.5)
    return image


def get_style_loss(features, targets):
    style_loss = tf.reduce_mean(tf.square(features, targets))
    return style_loss


def get_content_loss(features, targets):
    content_loss = 0.5 * tf.reduce_sum(tf.square(features, targets))
    return content_loss


# calculate gram matrix
def gram_matrix(input_tensor):
    " input_tensor is with shape [batch size, height, width, channel] "
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    # get the height and width of the input tensor
    input_shape = tf.shape(input_tensor)
    height = input_shape[1]
    width = input_shape[2]

    num_locations = tf.cast(height * width, tf.float32)
    scaled_gram = gram / num_locations

    return scaled_gram











