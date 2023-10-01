# Imports python modules
import argparse
import json
import ast
from PIL import Image
from torch.autograd import Variable
from torch import __version__
import argparse
import torch
import numpy as np
import os
from torch import nn,optim
from torchvision import models
from torchvision import datasets, transforms


os.chdir(r'D:\OneDrive - University of Moratuwa\Additioanl courses\Udacity\Ai_nanodegree\project2') #put your folder directory here

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('test_img', type = str, default = 'flowers/test/1/image_06752.jpg', 
                    help = 'Test img link')
    parser.add_argument('checkpoint', type = str, default = 'model.pth', 
                    help = 'Model saved link')
    parser.add_argument('--top_k', type = str, default = '5', 
                    help = 'Return top K most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Use a mapping of categories to real names')
    parser.add_argument('-gpu','--gpu', action='store_true' , help='Use GPU for inference')
    
    # you created with this function 
    return parser.parse_args()


def loading_model (file_path):
    
    checkpoint = torch.load (file_path) #loading checkpoint from a file
    model = models.vgg16(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    #call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode
    model.eval()
    
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
#     fig,ax=plt.subplots(2,2) #
    #resize only the width 256
#     ax[0][0].imshow(image)#
    width, height = image.size   # Get dimensions
    if (width>height):
        new_height = 256
        new_width  = round(new_height * width / height)
    else:
        new_width = 256
        new_height  = round(new_width * height / width)
    image = image.resize((new_width, new_height), Image.LANCZOS)
#     ax[0][1].imshow(image) #
    #center cut 224x224
    width, height = image.size   # Get dimensions
    left = round((width - 224)/2)
    top = round((height - 224)/2)
    x_right = round(width - 224) - left
    x_bottom = round(height - 224) - top
    right = width - x_right
    bottom = height - x_bottom
    
    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

#     ax[1][0].imshow(image) #
    #normalize
    np_img=np.array(image)/255

    #standardize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_img = (np_img - mean)/std
    return norm_img.transpose((2,0,1))

def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    model.eval()
    
    img=Image.open(image_path)

    img_arr = process_image(img)
    #we cannot pass image to model.forward as it is expecting tensor, not numpy array converting to tensor
    img = torch.from_numpy(img_arr).type(torch.FloatTensor)
    #used to make size of torch as expected. as forward method is working with batches,
    #doing that we will have batch size = 1
    img = img.unsqueeze (dim = 0) 
    img.to(device)
#     img_tens = img_tens.to(torch.device('cuda'))

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)
    output = torch.exp(output)
    #for the top k is 1 case just get to and print only the max value
    is_topk_1=False
    if topk==1:
        topk=2
        is_topk_1=True
    ps, idx = output.topk(topk,dim=1)
    ps = ps.numpy().squeeze() #converting both to numpy array
    idx = idx.numpy().squeeze()
    idx=idx.tolist()

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }
    
    idx=[mapping[i] for i in idx]
    idx=np.array(idx)


    return ps,idx,is_topk_1
#input image


in_arg = get_input_args()
# Use GPU if it's available
if (in_arg.gpu & torch.cuda.is_available()):
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
    
model=loading_model(in_arg.checkpoint)
model.to(device)
#bar plot for prediction
topk = int(in_arg.top_k)
ps,idx,is_topk_1=predict(in_arg.test_img,model,topk=topk)

ps=ps*100
with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
class_names = [cat_to_name [item] for item in idx]
if is_topk_1:
    print('{} with probability of {:.2f}%'.format(class_names[0],ps[0]))
else:
    for i in range(len(ps)):
        print('{} with probability of {:.2f}%'.format(class_names[i],ps[i]))