import cv2
import object_detection_model 
import config
from torchvision import transforms
import numpy as np
import os
import torch
import imutils
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import language
import openai
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
import re
from pytimedinput import timedInput
import random
import sys
openai.api_key = "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"



# load Lourve AD text dataset and create the information section for the prompt of original GPT3
def create_meta(objects, idx):
    meta = f"This is the {objects['Name'][idx]}. It is founded in {objects['Place'][idx]}. It is said to have been created in {objects['Year'][idx]}. "
    if not pd.isna(objects['Shape'][idx]):
        meta += f"Its' shape is {objects['Shape'][idx]}. "
    if not pd.isna(objects['Material'][idx]):
        meta += f"It is made of {objects['Material'][idx]}. "
    if not pd.isna(objects['Description'][idx]):
        meta += f"{objects['Description'][idx]}"
    return meta

# return two function used to retrieval object name and object information from the object label
def read_csv_museum():
    objects = pd.read_csv(config.PATH_CSV)
    label_to_name = {int(objects['Code'][idx]): objects['Name'][idx] for idx in range(len(objects))}
    label_to_meta = {int(objects['Code'][idx]): create_meta(objects, idx) for idx in range(len(objects))}
    return label_to_name, label_to_meta


# receive the information of object labels / visitor question -> process it to generate prompt -> call language to generate text
def tour_guide(video_label, question, times):
    prompt = str()
    if times == 0:
        return config.WELCOME 
    if not video_label:
        return

    if config.FINE_TUNE:
        if question:
            prompt = question
            prompt = question.replace("this object", f"the {label_to_name[video_label]} object (code {video_label})") 
            prompt = question.replace("it", f"the {label_to_name[video_label]} object (code {video_label})")  
        if times == 1:
            prompt = f"Describe the {label_to_name[video_label]} object (code {video_label})" 
        prompt = prompt + " ->"
        if prompt:
            completion = language.finetune(prompt)
            return completion
        else: return
    

    if not config.FINE_TUNE:
        context = f"Here is the information of the object (code {video_label}): {label_to_meta[video_label]}. Use the above information and your knowledge to answer this question: "
        if question:
            prompt = question
        if times == 1:
            prompt = f"Describe this object. " 
        if prompt:
            completion = language.davinci(context + prompt)
            return completion
        else: return
   


def object_detection(frame, detection_model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype="float32")
    image = transforms_ds(image)
    image = image.unsqueeze(0)
    (boxPreds, labelPreds) = detection_model(image)
    (startX, startY, endX, endY) = boxPreds[0]
    # determine the class label with the largest predicted probability
    labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
    i = labelPreds.argmax(dim=-1).cpu().item()
        
    label = str(config.idx_to_label[i])

    return (startX, startY, endX, endY), label

def draw_boxes(orig, bboxes, label):
    # Replace with your code to draw bounding boxes on frames
    startX, startY, endX, endY = bboxes
    orig = imutils.resize(orig, width=600)
    (h, w) = orig.shape[:2]
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(orig, str(label), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
    return orig

# use to find the object label in the visitor question
def find_int_in_str(string):
    # find all integers in the string
    integers = re.findall(r'\d+', string)
    if integers:
        return int(integers[0])


if __name__ == "__main__":
    detection_model = object_detection_model.init_model()
    detection_model.eval()
    transforms_ds = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=config.MEAN, std=config.STD)])
    label_to_name, label_to_meta = read_csv_museum()

    # WELCOME
    if config.IS_WELCOME:
        text = tour_guide(None, None, 0)
        language.text_to_speech(text)


    ls_demo_video = os.listdir(config.VIDEO_FOLDER_PATH)
    random.choice(ls_demo_video)

    # Loop
    while True:
        # CAPTURE VIDEO
        video_path = os.path.join(config.VIDEO_FOLDER_PATH, random.choice(ls_demo_video)) # take a random video from demo video folder
        cap = cv2.VideoCapture(video_path)
        label_ls=[]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
   
        # OBJECT DETECTION 
        #for i in range(total_frames):
        for i in range(config.PATIENCE):
            ret, frame = cap.read()
            if ret:
                orig = frame.copy()
                dim = (540, 960)
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                bboxes, label = object_detection(frame, detection_model)
                label_ls.append(label)
                orig = draw_boxes(orig, bboxes, label)
                cv2.imshow('output', orig)
                cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()
        

        # use the label with the highest frequency to be the label of the whole video
        freq = np.bincount(label_ls)
        video_label = np.argmax(freq)
    
        # time==0: create welcome text, time==1: create prompt of "Describe the <object name>", time>1: Q & A
        times = 1 
        question = str()

        # Q & A loop
        while True:
            # LANGUAGE MODEL
            text = tour_guide(video_label, question, times) # "Describe the <object name>"

            # TEXT TO SPEECH
            language.text_to_speech(text)
            times +=1    
            print()
            print()
            
            # SPEECH TO TEXT
            question, is_timed_out = timedInput("Do you want to ask anything... ", timeout=10)

            if is_timed_out: 
                break 
            else:
                
                integer = find_int_in_str(question) #check if the question has the object code

                if integer:
                    video_label = integer
                
                if question in ['end', 'exit', 'stop', 'goodbye']:
                    exit()

        