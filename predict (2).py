import numpy as np
import torchvision
import torch
import time
import random, os
import copy
import seaborn
import pandas
import json

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

from torch import nn, optim
from torch.autograd import Variable

import torch.nn.functional as F
from torch.optim import lr_scheduler
from collections import OrderedDict

from PIL import Image
import argparse

#set up parser
#add path to image
#add top k
#add save checkpoint
#add gpu inference
parser = argparse.ArgumentParser()
parser.add_argument("--imagepath", type = str, help = "enter the path to the image")
parser.add_argument("--topk", type = int, default = 5, help = """returns the top k classes where k is the your (e.g an input of 5 means the top 5 will be given)""")
parser.add_argument('--cat_to_names', type = str, help = 'Use a mapping of categories to real names')
parser.add_argument("--checkpoint", type = str, help = "path for the saved checkpoint to load it into predicting")
parser.add_argument("--gpu", type = str, default = "True", help = "CUDA computation enabling, faster training etc.")

args = parser.parse_args()
print('GPU on or off:', args.gpu)

#loading
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    #torch.load('my_file.pt', map_location=lambda storage, loc: storage)
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    if checkpoint['architecture'] == "vgg19":
        model = models.vgg19(pretrained = True)
    elif checkpoint['architecture'] == "densenet121":
        model = models.densenet121(pretrained = True) 
    
    epochs = checkpoint['epochs']
    
    #after running into issues regarding a keyerror for checkpoint['model'], I removed it because reading the code
    #it has no purpose, hence I decided to comment it out and after adding one more thing to my code it removed my errors
    #and everything works as it should (GPU on or off, both work now)
    #model = checkpoint['model']
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model

checkpoint = args.checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #recommended by mentor help
model = load_checkpoint(checkpoint)
model.to(device) #recommended by mentor help

# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    #this will scale my images propely and return a numpy array
    image = Image.open(image)
    transforming_image = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    image = np.array(transforming_image(image))
    
    #spent time debugging this, realised in the end that this requires a pytorch tensor, before I was returning the image on 
    #its own, torch.from_numpy(image) fixed my error
    return torch.from_numpy(image)



def class_predict(image_path, model, topk=5):  
    model.to('cpu')
    #model.class_to_idx = training_data.class_to_idx
    image_torch = Image.open(image_path) 
    image_torch = process_image(image_path)
    image_torch = image_torch.unsqueeze_(0)

    model.eval()
    with torch.no_grad():
        #I did this to fix an error which asked to return a float_tensor
        image_torch = image_torch.float()
        #changed to model.forward(image_torch) from model.forward(image_path), mentor help suggested this as the reply to
        #my query. It is because I had tried to set logps to string in image_path when it needs a torch function as its
        #using model.forward
        logps = model.forward(image_torch.float())
        ps = torch.exp(logps)
        
        top_k_probs, top_class = ps.topk(topk, dim=1)
        # Get the first items in the tensor list to get the list of probs
        top_k_probs = top_k_probs.tolist()[0]

        
        #reversed the mapping, mentor help suggested my class_predict function was incomplete.
        #when re-reading the above help, it says to make sure I invert the mapping from idx to class.
        model.idx_to_class = dict(map(reversed, model.class_to_idx.items()))
        
        #found this tip from mentor help as I was struggling to gather classes. I implemented it so it would also get the first
        #items in the tensor list, as I had an array within the tensor, hence the top_class[0].tolist() when looping through
        classes = [model.idx_to_class[idx] for idx in top_class[0].tolist()]
    
    return top_k_probs, classes

image_path = args.imagepath
top_k = args.topk
topprobs, classes = class_predict(image_path, model, top_k)

#reviwer told me to have cat_to_name load in via terminal, so what I've done is an if statement so nothing shows
#UNTIL user inputs --cat_to_names cat_to_name.json, only then do we get the output of the flower names. Really simple change, will check with mentor to
#see if this works before second submission
if args.cat_to_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]
    print(names)
    

print(classes)
print(topprobs)


        
 
        



                    