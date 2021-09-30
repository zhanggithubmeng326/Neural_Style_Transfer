"""" implementation of neural style transfer using the VGG19 model as feature extractor"""

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as k
from IPython.display import display as display_fn
from IPython.display import Image, clear_output
import utils

# choose corresponding layers of content and style images
content_layers = ['conv2d_88']

style_layers = ['conv2d',
                'conv2d_1',
                'conv2d_2',
                'conv2d_3',
                'conv2d_4']

total_layers = content_layers + style_layers
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# build the model of feature extractor using inception model
def feature_extractor(layers):
    # load Inception V3 with imagenet weights and without fully connected layer at the top of the network
    inception= tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # freeze the weights of the model
    inception.trainable = False

    # create a list of layers
    output_layers = [inception.get_layer(name).output for name in layers]

    # create model
    model = tf.keras.Model(input_dim=inception.input, outputs=output_layers)

    return model


# create an instance of feature extractor
feature_extractor = feature_extractor(total_layers)


# get the features of style images
def get_style_image_features(image):
    preprocessed_style_image = utils.preprocess_image(image)
    outputs = feature_extractor(preprocessed_style_image)
    style_outputs = outputs[:num_style_layers]
    gram_style_features = [utils.gram_matrix(style_layer) for style_layer in style_outputs]

    return gram_style_features


# get the features of content images
def get_content_image_features(image):
    preprocessed_style_image = utils.preprocess_image(image)
    outputs = feature_extractor(preprocessed_style_image)
    content_outputs = outputs[num_style_layers:]

    return content_outputs


# calculate the total loss
def get_style_content_loss(style_outputs,
                           style_targets,
                           content_outputs,
                           content_targets,
                           style_weight,
                           content_weight):

    style_loss = tf.add_n([utils.get_style_loss(style_output, style_target) for style_output, style_target in zip(style_outputs, style_targets)])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([utils.get_content_loss(content_output, content_target) for content_output, content_target in zip(content_outputs, content_targets)])
    content_loss *= content_weight / num_content_layers

    loss_total = style_loss + content_loss

    return loss_total


def calculate_gradients(image, style_targets, content_targets, style_weight, content_weight):
    with tf.GradientTape() as tape:
        style_features = get_style_image_features(image)
        content_features = get_content_image_features(image)
        loss = get_style_content_loss(style_features, style_targets, content_features, content_targets, style_weight, content_weight)
    gradients = tape.gradient(loss, image)

    return gradients


# generate the stylized image
def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4,
                       optimizer_name='adam', epochs=1, steps_per_epoch=1):
    images = []
    step = 0
    style_targets = get_style_image_features(style_image)
    content_targets = get_content_image_features(content_image)

    # initialize the generated image features
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image)
    images.append(content_image)

    for i in range(epochs):
        for j in range(steps_per_epoch):
            step += 1
            gradients = calculate_gradients(generated_image, style_targets, content_targets, style_weight,
                                            content_weight)
            optimizer = tf.optimizers.Adam(
                tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=80.0, decay_rate=0.80,
                                                               decay_steps=100))
            optimizer.apply_gradients([(gradients, generated_image)])
            generated_image.assign(utils.clip_image_values(generated_image, min_value=0.0, max_value=255.0))
            print(".", end='')

        # display the current stylized image
        clear_output(wait=True)
        display_image = utils.tensor_to_image(generated_image)
        display_fn(display_image)

        images.append(generated_image)
        print('epoch: {}'.format(i))

        # convert to uint8 (expected dtype for images with pixels in the range [0,255])
        generated_image = tf.cast(generated_image, dtype=tf.uint8)

        return generated_image, images


if '__name__' == '__main__':
    content_path = '/home/mzhang/Style_Transfer/content.png'
    style_path = '/home/mzhang/Style_Transfer/style.png'

    # display the content and style image
    content_image, style_image = utils.load_images(content_path, style_path)
    utils.show_images_with_objects(images=[content_image, style_image], titles=['content.png', 'style.content'])

    STYLE_WEIGHT = 1
    CONTENT_WEIGHT = 1e-32
    EPOCHS = 20
    STEPS_PER_EPOCH = 100

    styled_image, display_images = fit_style_transfer(style_image=style_image, content_image=content_image,
                                                      style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT,
                                                      epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)


