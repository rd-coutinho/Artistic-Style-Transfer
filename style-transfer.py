# -*- coding: utf-8 -*-
"""
@author: renat
"""

# Define Image Processing utilities

import numpy as np
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array


def preprocess_image(image_path, height=None, width=None):
    height = 400 if not height else height
    width = width if width else int(width * height / height)
    img = load_img(image_path, target_size=(height, width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Setup symbolic placeholders for images
from keras import backend as K

# This is the path to the image you want to transform.
TARGET_IMG = 'data/italy_street.jpg'
# This is the path to the style image.
REFERENCE_STYLE_IMG = 'data/style1.png'

width, height = load_img(TARGET_IMG).size
img_height = 320
img_width = int(width * img_height / height)


target_image = K.constant(preprocess_image(TARGET_IMG, height=img_height, width=img_width))
style_image = K.constant(preprocess_image(REFERENCE_STYLE_IMG, height=img_height, width=img_width))

# Placeholder for our generated image
generated_image = K.placeholder((1, img_height, img_width, 3))

# Combine the 3 images into a single batch
input_tensor = K.concatenate([target_image,
                              style_image,
                              generated_image], axis=0)

# Setup the VGG-19 CNN model
# We will build the VGG19 network with our batch of 3 images as input.
# The model will be loaded with pre-trained ImageNet weights and no fully connected layers.
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
layers = dict([(layer.name, layer.output) for layer in model.layers])
#model.summary()

# Define the content loss
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# Define the style loss
def style_loss(style, combination, height, width):
    
    def build_gram_matrix(x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram_matrix = K.dot(features, K.transpose(features))
        return gram_matrix

    S = build_gram_matrix(style)
    C = build_gram_matrix(combination)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# Define the total variation loss
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# Building the loss function
# weights for the weighted average loss function
content_weight = 0.025
style_weight = 1.0
total_variation_weight = 1e-4

## Set the content and style layers based on VGG architecture
# define function to set layers based on source paper followed
def set_cnn_layers(source='gatys'):
    if source == 'gatys':
        # config from Gatys et al.
        content_layer = 'block5_conv2'
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                        'block4_conv1', 'block5_conv1']
    elif source == 'johnson':
        # config from Johnson et al.
        content_layer = 'block2_conv2'
        style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 
                        'block4_conv3', 'block5_conv3']
    else:
        # use Gatys config as the default anyway
        content_layer = 'block5_conv2'
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                        'block4_conv1', 'block5_conv1']
    return content_layer, style_layers

# Set the source research paper followed and set the content and style layers
source_paper = 'gatys'
content_layer, style_layers = set_cnn_layers(source=source_paper)

## Build the weighted loss function
# initialize total loss
loss = K.variable(0.)

# Add content loss
layer_features = layers[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
#loss += content_weight * content_loss(target_image_features, combination_features)
loss = loss + (content_weight * content_loss(target_image_features, combination_features))

# Add style loss
for layer_name in style_layers:
    layer_features = layers[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features, 
                    height=img_height, width=img_width)
    loss += (style_weight / len(style_layers)) * sl

# Add total variation loss
loss += total_variation_weight * total_variation_loss(generated_image)

# Build the optimizer
# Get the gradients of the generated image wrt the loss
grads = K.gradients(loss, generated_image)[0]

# Function to fetch the values of the current loss and the current gradients
fetch_loss_and_grads = K.function([generated_image], [loss, grads])


class Evaluator(object):

    def __init__(self, height=None, width=None):
        self.loss_value = None
        self.grads_values = None
        self.height = height
        self.width = width

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.height, self.width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator(height=img_height, width=img_width)

# Run the optimizer

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

this_one = TARGET_IMG.split('.')[0]
this_one = this_one.split('/')[1]
result_prefix = 'style_transfer_result_'+this_one
result_prefix = result_prefix+'_'+source_paper
iterations = 20

# Run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss.
# This is our initial state: the target image.
# Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
x = preprocess_image(TARGET_IMG, height=img_height, width=img_width)
x = x.flatten()

start = time.time()
for i in range(iterations):
    print('Start of iteration', (i+1))
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    if (i+1) % 5 == 0 or i == 0:
        # Save current generated image only every 5 iterations
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        fname = result_prefix + '_at_iteration_%d.png' %(i+1)
        imsave(fname, img)
        print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i+1, end_time - start_time))

end = time.time()
print('Tempo total de processamento: {} minutos'.format((end - start)/60))
    
# View Results

from skimage import io
from glob import glob
from matplotlib import pyplot as plt

#%matplotlib inline

# Example 1
cr_content_image = io.imread('data/city_road.jpg')
cr_style_image = io.imread('data/style2.png')
cr_iter1 = io.imread('style_transfer_result_city_road_gatys_at_iteration_1.png')
cr_iter10 = io.imread('style_transfer_result_city_road_gatys_at_iteration_10.png')
cr_iter20 = io.imread('style_transfer_result_city_road_gatys_at_iteration_20.png')

fig = plt.figure(figsize = (12, 4))
ax1 = fig.add_subplot(1,2, 1)
ax1.imshow(cr_content_image)
t1 = ax1.set_title('City Road Image')
ax2 = fig.add_subplot(1,2, 2)
ax2.imshow(cr_style_image)
t2 = ax2.set_title('Edtaonisl Style')

fig = plt.figure(figsize = (20, 5))
ax1 = fig.add_subplot(1,3, 1)
ax1.imshow(cr_iter1)
t1 = ax1.set_title('Iteration 1')
ax2 = fig.add_subplot(1,3, 2)
ax2.imshow(cr_iter10)
t2 = ax2.set_title('Iteration 10')
ax3 = fig.add_subplot(1,3, 3)
ax3.imshow(cr_iter20)
t3 = ax3.set_title('Iteration 20')
t = fig.suptitle('City Road Image after Style Transfer')