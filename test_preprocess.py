# Importing all necessary libraries 
import cv2 
import os 
import glob
import re
import os, numpy, PIL
from PIL import Image
from torchvision.transforms import Compose, Resize,  CenterCrop
from matplotlib import cm

"""
def test_transform(image_size):
    return np_transforms.Compose([
        np_transforms.CenterCrop(size=(1080, 1080)),
        #np_transforms.Scale(size=256)
    ])
"""
def test_transform(image_size):
    return Compose([
        Resize(image_size),
        CenterCrop(image_size)
    ])

def vid2frame(path):
    # Read the video from specified path 
    video_names = glob.glob(path+ '/*')
    transform = test_transform(256)

    for video_path in video_names:
        video_name = video_path.split('/')[-1]
        video_name = video_name.split('.')[0]

        try:
        # creating a folder
            folder2save_frames = os.path.join(path, video_name)
            if not os.path.exists(folder2save_frames): 
                os.makedirs(folder2save_frames) 
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data')
        
        cam = cv2.VideoCapture(video_path)
        video_name = video_name + '.avi'
        out = cv2.VideoWriter(os.path.join(folder2save_frames, video_name),cv2.VideoWriter_fourcc(*'DIVX'), 30, (256,256))

        # frame 
        currentframe = 0

        while(True): 
            
            # reading from frame 
            ret,frame = cam.read() 
        
            if ret: 
                # if video is still left continue creating images 
                name = 'frame' + str(currentframe) + '.jpg'
                name = os.path.join(folder2save_frames, name)
                print ('Creating...' + name)
                fram_pil = Image.fromarray(frame)
                frame = transform(fram_pil)
                frame = numpy.array(frame)
                # writing the extracted images 
                cv2.imwrite(name, frame)
                out.write(numpy.uint8(frame))
        
                # increasing counter so that it will show how many frames are created 
                currentframe += 1
            else: 
                break
        
        # Release all space and windows once done 
        cam.release()
        out.release()
        cv2.destroyAllWindows()
        calc_mean(folder2save_frames)

    return
    
def calc_mean(video_path):
    video_name = video_path.split('/')[-1]
    #video_name = video_name.split('.')[0]
    
    # Access all PNG files in directory
    allimages=os.listdir(video_path)
    imlist = [os.path.join(video_path, x) for x in allimages if x[-4:] in [".png",".PNG", ".jpg", "JPG"]]
    imlist.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Assuming all images are the same size, get dimensions of first image
    w,h=Image.open(imlist[0]).size
    N=len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr=numpy.zeros((h,w,3),numpy.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr=numpy.array(Image.open(im),dtype=numpy.float)
        arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

    # Generate, save and preview final image
    out=Image.fromarray(arr,mode="RGB")
    #out.show()
    name = video_name + '_Average.jpg'
    name = os.path.join(video_path, name)
    out.save(name)

    return

if __name__ == '__main__':
    path = "/Users/Maayan/Documents/databases/test2"
    vid2frame(path)
