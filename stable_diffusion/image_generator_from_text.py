import time
import PIL
import tensorflow as tf
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse
import numpy as np
from upscaleImage import upscale_image
import random

# simple function to load image
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# simple function to display image


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def save_images(images, titles):
    for i in range(len(images)):
        save_image(images[i], titles[i])


def save_image(image, title):
    print("save_image() with title of {} and image shape of {}".format(title, image.shape))
    if len(image.shape) == 4:
        tf.keras.utils.save_img(title, np.squeeze(image, 0)) 
    elif len(image.shape) == 3:
        tf.keras.utils.save_img(title, image)

def plot_images(images, titles):
    print("images length: ", len(images))
    if len(images) > 1:
        nrows, hop_size = adjustGrid(len(images))
        plt.figure(figsize=(hop_size*10, nrows*10))
        plt.margins(0.0)
        # for each lines of the plo%
        print("plotting {} total images with {} columns in {} rows".format(
            len(images), hop_size, nrows))
        for i in range(len(images)):
            ax = plt.subplot(nrows, hop_size, i+1)
            plt.imshow(images[i])
            plt.axis("off")
            plt.savefig(titles[i] + str(i) + "_figure.png")
    else:
        plot_image(images)


def plot_image(image):
    image = np.squeeze(image, 0)
    plt.figure(figsize=(20, 20))
    plt.margins(0.0)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(output_name + str(img_num) + "_figure.png")


def singleImage(source, style, hub_model):
    output_images = []
    source_image = load_img(source)
    style_image = load_img(style)
    output_images.extend(hub_model(tf.constant(
        source_image), tf.constant(style_image))[0])
    save_images(output_images)
    output_images.extend(style_image)
    output_images.extend(source_image)
    output_images.reverse()
    plot_images(output_images)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def adjustGrid(num, cmax=5):
    # find place where we "flip", use that to determine number of rows and columns
    # go from 1 to half of number
    rnum = 1
    cnum = num
    for rn in range(1, ceil(num/2) + 1):
        rnum = rn
        cnum = ceil(num/rnum)
        # if the number of columns is no longer greater than the number of rows, undo the change and export numbers
        if cnum <= rnum and cnum <= cmax:
            break
    total = rnum * cnum
    # if total < num:
    # rnum = rnum + 1
    return (rnum, cnum)

def getBatchString(current_idx, batch_size):
    num_digits = len(str(batch_size))
    return "{:0{}}".format(current_idx, num_digits)

def generateOutputNames(prompts, batch_size, seed):
    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    path = os.getcwd() + "/output_images/"
    prompt_names = []
    print("prompts are {}".format(prompts))
    for name in prompts:
        for batch in range(batch_size):
            _name = time_str + name.replace(' ', '_')[:40] + "_s" + str(seed) + "_"
            b_string = getBatchString(batch, batch_size)
            prompt_names.append(path + _name + str(batch) + ".png")
            print("image name generated: {}".format(prompt_names[-1]))
    return prompt_names

def createModel(img_width, img_height, jit_compile=True):
    model = keras_cv.models.StableDiffusion(
        img_width=img_width, img_height=img_height, jit_compile=jit_compile)
    return model

def main(arg_dict):
    # create a basic naming root for any files that we create and want to save during this session
    start_time = time.time()
    output_width = arg_dict["output_width"]
    output_height = arg_dict["output_height"]
    prompts = arg_dict["prompts"]
    batch_size = arg_dict["batch_size"]
    steps = arg_dict["steps"]
    seed = arg_dict["seed"]
    plot_output = arg_dict["plot_output"]
    upscale_factor = arg_dict["upscale"]


    model = createModel(output_width, output_height)
    # for automatically labling each output file uniquely
    print("prompts are: {}".format(prompts))
    output_names = generateOutputNames(prompts, batch_size, seed)
    print("output names: {}".format(output_names))
    idx = 0
    #########################################################
    for prompt in prompts:
        for batch in range(batch_size):
            print("creating batch # {} for prompt {}".format(batch, prompt))
            image = model.text_to_image(
                prompt,
                seed=seed,
                num_steps=steps,
                batch_size=1
            )      
            save_image(image, output_names[idx])
            if upscale_factor != 1.0:
                upscale_image(output_names[idx], upscale_factor)
            idx += 1
    
        keras.backend.clear_session()  # Clear session to preserve memory

    if (plot_output):
        plot_images(image)
      
    end_time = time.time()
    runtime = end_time - start_time
    print("Total function runtime is {} with {} as parameters".format(runtime, arg_dict))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_width", type=int,
                        help="output width of generated image in pixels", default=256)
    parser.add_argument("--output_height", type=int,
                        help="output height of generated image in pixels", default=256)
    parser.add_argument("--prompts", nargs="+", type=str, help="list of prompts for the program to generate images based on",
                        default="happy black cat floating on clouds of love")
    parser.add_argument("--batch_size", type=int,
                        help="how many images to generate", default=4)
    parser.add_argument(
        "--steps", type=int, help="number of steps contained in each epoch", default=100)
    parser.add_argument("--plot_output", type=bool, help="if set to true, program will plot the output images", default=False)
    parser.add_argument(
        "--seed", type=int, help="seed for random number generator, for producing consistant results", default=random.randint(0, 99999))    
    parser.add_argument("--upscale", type=float, default=1.0, help="After generating the images they will be upscaled to this factor")
    args = vars(parser.parse_args())

    main(args)
