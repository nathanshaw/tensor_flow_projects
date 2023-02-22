import tensorflow as tf
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import IPython.display as display
import PIL.Image
from math import ceil, floor
from datetime import datetime
import argparse

def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries.
    shift = tf.random.uniform(
        shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled

# simple function to load image


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    max_dim = 512
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def save_images(images):
    global img_num
    for i in range(len(images)):
        tf.keras.utils.save_img(output_name + str(img_num) + ".png", images[i])
        img_num = img_num + 1


def save_tf_image(image, added_text=""):
    global img_num
    tf.keras.utils.save_img("{}{}{}.png".format(
        output_name, str(img_num), added_text), image)
    img_num = img_num + 1


def plot_images(images):
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
            plt.savefig(output_name + str(img_num) + "_figure.png")
    else:
        plot_image(images)


def plot_image(image):
    image = np.squeeze(image, 0)
    plt.figure(figsize=(20, 20))
    plt.margins(0.0)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(output_name + str(img_num) + "_figure.png")


def singleImage(source, style):
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

# Download an image and read it into a NumPy array.


def loadImage(path, max_dim=None):
    new_shape = ()
    img = PIL.Image.open(path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    img = np.array(img)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    # now finally reduce the size of the image
    img = tf.image.resize(img, new_shape)
    return np.array(img)

# Normalize an image


def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

# Display an image


def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))


def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)


class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[2], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[y:y+tile_size, x:x+tile_size]
                    loss = calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01,
                                octaves=range(-2, 3), save_modulo=10, octave_scale=1.3, save_all=False):
    base_shape = tf.shape(img)
    img = tf.keras.utils.img_to_array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    initial_shape = img.shape[:-1]
    img = tf.image.resize(img, initial_shape)
    epoch_num = 0
    for octave in octaves:
        # Scale the image based on the octave
        new_size = tf.cast(tf.convert_to_tensor(
            base_shape[:-1]), tf.float32)*(octave_scale**octave)
        new_size = tf.cast(new_size, tf.int32)
        img = tf.image.resize(img, new_size)
        if octave == octaves[-1]:
            print("doubing the steps for final octave")
            steps_per_octave = steps_per_octave * 2
        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)
            sstr = ""
            if step < 10:
                sstr = "000" + str(step)
            elif step < 100:
                sstr = "00" + str(step)
            elif step < 1000:
                sstr = "0" + str(step)

            if step % save_modulo == 0:
                save_tf_image(img, "_{}".format(epoch_num * steps_per_octave + step, sstr))
                # display.clear_output(wait=True)
                # show(deprocess(img))
                print("Octave {}, Step {} image saved".format(octave, step))
        epoch_num = epoch_num + 1

    result = deprocess(img)
    return result


parser = argparse.ArgumentParser(
    prog="Command Line Deep Dreamer",
    description="automatically conducts octave deep dreaming on all images in source_images directory"
)
parser.add_argument('-sf', '--save_frequency', default=10, type=int,
                    help='modulo value for how often images of the dream process will be shared, default value is 10, \
                    meaning one out of every 10 training images will be saved')

parser.add_argument('-os', '--octave_scale', default=1.3, type=float,
                    help='how much is the size of the image changed between octaves? default is 1.3')

parser.add_argument('-spo', '--steps_per_octave', default=80, type=int,
                    help='how many steps are conducted for each Octave? Default value is 80')

parser.add_argument('-ss', '--step_size', default=0.01, type=float,
                    help='how large are the steps? default value of 0.01, larger values cause faster dreaming')
# positional argument
parser.add_argument('-cv', '--create_videos', default=False,  type=bool,
                    help="if set to true will use output images to create .mp4 file of training process")

parser.add_argument('-md', '--max_dim', choices=range(64, 1920), default=512, type=int,
                    help="maximum dimension (x or y axis) for the output image, accepts values between 64 and 1920 with a default of 512")

# this needes to be a list of ints
parser.add_argument('-ol', '--octaves', default="-2, -1, 0, 1, 2", type=str,
                    help='list of ints representing which octaves program will run through, default is "-2, -1, 0, 1, 2"')

# this needes to be a list of ints
parser.add_argument('-mode', '--mode', default="frost", type=str, choices=["frost", "eyes", "standard", "test"],
                    help='what layers from the model will be used for generating the image, default is "standard" while other options\
                        include "test", "dim_blur", "waves", "ripples", "hair", "spirals", frost", "eyes", "pools" and "pools2"')

if __name__ == "__main__":
    args = parser.parse_args()
    print("command line arguments parsed, {}".format(args))

    source_img_paths = os.listdir(os.getcwd() + "/source_images/")
    source_img_paths = [os.getcwd() + "/source_images/" +
                        x for x in source_img_paths if x.lower().endswith('.jpg') or x.lower().endswith('.png')]

    print("{} source image paths : ".format(len(source_img_paths)),
          [x[-20:] for x in source_img_paths])

    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    output_name = os.getcwd() + "/output_images/" + time_str
    global img_num
    img_num = 0
    print("output name is: {}".format(output_name))

    base_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet',
        classifier_activation='softmax',
        )

    # Maximize the activations of these layers
    # names = ['mixed3', 'mixed5']
    print(base_model.summary())
    if args.mode == "frost":
        names = ['mixed5']
    elif args.mode == 'eyes':
        names = ['mixed3']
    elif args.mode == 'standard':
        names = ['mixed3', 'mixed5']
    elif args.mode == 'dim_blur':
        names = ['mixed7']
    elif args.mode == 'waves':
        names = ['mixed1']
    elif args.mode == 'ripples':
        names = ['mixed2']
    elif args.mode == 'hair':
        names = ['mixed4']
    elif args.mode == 'spiral':
        names = ['mixed6']
    elif args.mode == 'pools':
        names = ['mixed8']
    elif args.mode == 'pools2':
        names = ['mixed9']
    elif args.mode == 'test':
        names = ['mixed2', 'mixed3', 'mixed4']
    
    layers = [base_model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    dream_imgs = []
    get_tiled_gradients = TiledGradients(dream_model)

    octave_list = [int(i) for i in args.octaves.split(',')]
    print("octave_list : {}".format(octave_list))
    # now to complete the loop, lets grab all of our source images and run the full program on them =)
    # now to complete the loop, lets grab all of our source images and run the full program on them =)
    final_images = []
    for path in source_img_paths:
        img = loadImage(path, max_dim=args.max_dim)
        final_images.append(
            run_deep_dream_with_octaves(
                img=img, step_size=args.step_size,
                save_modulo=args.save_frequency,
                steps_per_octave=args.steps_per_octave,
                octave_scale=args.octave_scale,
                octaves=octave_list
            ))
    save_images(final_images)
    # last step should be to run the create_mp4_from_imgs script
    if args.create_videos is True:
        os.system('python create_mp4_from_images.py')
    print("dream_source_images.py has successfully completed running =)")
    print("------------------------------------------------------------")