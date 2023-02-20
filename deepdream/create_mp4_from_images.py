import os
import cv2 as cv
from datetime import datetime
import natsort

global vid_num

def createOutputFileName(custom_text=''):
    global vid_num
    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    output_file_name = "{}video_output_{}_{}.mp4".format(os.getcwd(), time_str, custom_text, vid_num)
    vid_num = vid_num + 1
    return output_file_name

# Determine the width and height from the first image
# combine images into a .mov file
def createVideo(source_paths, output_path, shape):
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv.VideoWriter(output_path, fourcc, 20.0, (shape[1], shape[0]))
    for i in range(len(source_img_paths)):
        frame = cv.imread(source_paths[i])
        if frame.shape == shape:
            out.write(frame) # Write out frame to video
            if (cv.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                break
    # Release everything if job is finished
    out.release()
    cv.destroyAllWindows()

def removeShapeFromList(shape, lst):
    new = []
    for l in lst:
        frame = cv.imread(l)
        if frame.shape != shape:
            new.append(l)
    return new

def getImageShape(path):
    frame = cv.imread(path)
    return frame.shape

if __name__ == "__main__":
    print("Starting program create_mp4_from_images")
    target_shapes = []
    source_img_paths = os.listdir(os.getcwd() + "/output_images/")
    source_img_paths = [os.getcwd() + "/output_images/" + x for x in source_img_paths if x.lower().endswith('.jpg') or x.lower().endswith('.png')]
    source_img_paths = natsort.natsorted(source_img_paths)

    print("{} starting sources".format(len(source_img_paths)))
    while len(source_img_paths) > 0:
        vid_num = 0
        shape = getImageShape(source_img_paths[0])
        target_shapes.append(shape) 
        output_mov_path = createOutputFileName(str(shape[1]) + "_" + str(shape[0]))
        createVideo(source_img_paths, output_mov_path, target_shapes[-1])
        source_img_paths = removeShapeFromList(target_shapes[-1], source_img_paths)
        print("The output video is {}".format(output_mov_path))
        print('{} remaining sources:'.format(len(source_img_paths)))
    print("Ending program create_mp4_from_images")
    print("-"*20)