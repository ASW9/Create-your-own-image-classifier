from collections import OrderedDict
import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms, datasets, models
import argparse
import numpy
import json

#setting up my argument parsers
parser = argparse.ArgumentParser()
parser.add_argument("--directory", help = "file directory")
parser.add_argument("--architecture", default = "vgg19", help ="neural net architecture either vgg19 or densenet121")
parser.add_argument("--epochs", type = int, default = 1, help = "number of epochs")
parser.add_argument("--learningrate", type = float, default = 0.001, help = "learning rate")
parser.add_argument("--hiddenunits", type = int, default = 512, help = "integer value of the hidden units")
parser.add_argument("--gpu", action = "store_true", default = True, help = "CUDA computation enabling, faster training etc.")

#loading my parser values into args
args = parser.parse_args()
#printing wether or not gpu is on to test
print('GPU on or off:', args.gpu)


print("loading the data:")

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# TODO: Define your transforms for the training, validation, and testing sets
training_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


validation_transform = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

testing_transform  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder

""""define batch size --> NOTE FOR SELF: keeping at 64 incase I train with cpu to avoid the neural network crashing. 
                                         If using only gpu, I can increase batch size to 128 for e.g or 256 etc.""" 
batch_s = 64
#define your training, validation and testing data (loading the datasets with ImageFolder)
training_data = datasets.ImageFolder(train_dir , transform = training_transform)
validation_data = datasets.ImageFolder(valid_dir , transform = validation_transform)
testing_data = datasets.ImageFolder(test_dir,transform = testing_transform )

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(training_data, batch_size = batch_s, shuffle = True)

validloader = torch.utils.data.DataLoader(validation_data, batch_size = batch_s, shuffle = True)

testloader = torch.utils.data.DataLoader(testing_data, batch_size = batch_s, shuffle = True)
 
with open("cat_to_name.json", 'r') as f:
    cat_to_name = json.load(f)                    

print("End of data loader")    
# TODO: Build and train your network

#defining device incase GPU isn't available, taken from my work on 
#part 8 - transfer learning exercise from the udacity curriculum
print("Building model...")



#define model - using vgg19
if args.architecture == "vgg19":
    # Freeze the parameters to avoid backpropogation
    #(taken from my work on part 8 - transfer learning exercise from the udacity curriculum)
    model = models.vgg19(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    #defining new untrained neural net which is feedforward, using ReLU functions and LogSoftmax for my output
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 2048)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(2048, 256)),
                              ('relu', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.2)),
                              ('fc3', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
elif args.architecture == "densenet121":
    model = models.densenet121(pretrained = True)
    # Freeze the parameters to avoid backpropogation
    #(taken from my work on part 8 - transfer learning exercise from the udacity curriculum)
    for param in model.parameters():
        param.requires_grad = False
    #defining new untrained neural net which is feedforward, using ReLU functions and LogSoftmax for my output
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 2048)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(2048, 256)),
                              ('relu', nn.ReLU()),
                              ('dropout2', nn.Dropout(p=0.2)),
                              ('fc3', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))


   
hidden_units = args.hiddenunits
learning_rate = args.learningrate
#(taken from my work on part 8 - transfer learning exercise from the udacity curriculum)
model.classifier = classifier
#defining criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("architecture type{}".format(args.architecture))

print("model training...")

epochs = args.epochs
steps  = 0
running_loss = 0
print_every  = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        #training step for my model
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss  = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        #validation step (this is where I previously went wrong, didn't understand the need or anything for this part now 
        #I understand that this isnt part of the testing stage, it should be done with the training after creating the 
        #untrained model)
        #Got this wrong when asking my mentor help question, so use this new understanding to correctly implement validation 
        #loop
        
        #UPDATE FOR SELF: Implemented correctly
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}... "
                  f"Train loss: {running_loss/print_every:.3f}... "
                  f"Validation loss: {test_loss/len(validloader):.3f}... "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

print("model training completed...")
print("...testing the model")

accuracy = 0
test_loss = 0
model.eval()
for images, labels in testloader:
    
    images, labels = images.to(device), labels.to(device)
    logps = model(images)
    loss = criterion(logps, labels)
    test_loss += loss.item()
                
    ps = torch.exp(logps)
    top_ps, top_class = ps.topk(1, dim =1)
    equality = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()


print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")
running_loss = 0
model.train()


print("saving checkpoint")
model.class_to_idx = training_data.class_to_idx
arch = args.architecture
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'learning_rate': 0.01,
              'model' : models.vgg19(pretrained = True),
              'epochs': epochs,
              'batch_size': batch_s,
              'classifier': model.classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'architecture': arch
             }
   
torch.save(checkpoint, 'checkpoint.pth')



                  
           
                    
                   
    
