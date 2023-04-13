import argparse
import os

import tensorflow as tf
from tensorflow import keras

def make_upscale_model(input_shape, output_shape):
    print("generating upscale model with input shape of {} and output shape of {}".format(input_shape, output_shape))
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_shape[0], input_shape[1], 3)))
    model.add(keras.layers.Lambda(lambda x: tf.image.resize(x, output_shape)))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss="ssim")
    return model


def upscale_image(path, scale):
    # Load image
    print("loading image at path: {}".format(path))
    img = keras.preprocessing.image.load_img(path)
    print("img:", img)
    print("img type: {}".format(type(img)))
    # Calculate target image size based on scale
    width, height = img.size
    target_size = (int(width * scale), int(height * scale))

    # Create model to upscale image
    model = make_upscale_model((width, height), target_size)

    # Upscale image
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    upscaled_array = model.predict(img_array)
    upscaled_img = keras.preprocessing.image.array_to_img(upscaled_array[0])

    # Save upscaled image
    output_path = os.path.splitext(path)[0] + "_upscaled.png"
    upscaled_img.save(output_path)

    print("Image saved to:", output_path)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Upscale an image using a neural network")
    parser.add_argument("--path", type=str, help="path to the image file")
    parser.add_argument("--scale", type=float, help="upscaling ratio")

    # Parse command-line arguments
    args = parser.parse_args()

    # Upscale image
    upscale_image(args.path, args.scale)
