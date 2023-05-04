#torch.device()
DEVICE = "cpu"  

#number of objects in the dataset
NUM_CLASSES = 9  

# path to object detection model checkpoint
CKPT_PATH = "/Users/cong/Downloads/0_ML709_submit/ckpt/detector_efficient_b0.pth" 

# encode label
label_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 10: 7, 15: 8}

# decode label
idx_to_label = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 15}

# image size
IMG_HEIGHT = 960
IMG_WIDTH = 540

# mean and std of ImageNet dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# use fine_tune or original GPT3
FINE_TUNE = False

# welcome text
WELCOME = "Hello and welcome to Louvre Abu Dhabi! My name is Voxie and I will be your tour guide today. I am very excited to show you all the beautiful art and artifacts that this museum has to offer. As a tour guide, my goal is to provide you with an informative and enjoyable experience, so please feel free to ask me any questions you may have throughout our tour."

# path to the csv file of Lourve Abu Dhabi text dataset
PATH_CSV = '/Users/cong/Downloads/0_ML709_submit/Museum-Object.csv'

# path to demo video folder
VIDEO_FOLDER_PATH = '/Users/cong/Downloads/0_ML709_submit/demo-video'
#VIDEO_PATH = '/Users/cong/Downloads/0_ML709/demo-video/IMG_2741.MOV'

# after {PATIENCE} frames, the system will move to the Q&A section
# uses to prevent the case that the input video is too long.
PATIENCE = 60
# do you want the model to greet you when you run the code
IS_WELCOME = True