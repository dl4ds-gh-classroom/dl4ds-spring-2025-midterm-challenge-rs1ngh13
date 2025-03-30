import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm  # For progress bars
from itertools import product
import matplotlib.pyplot as plt
import eval_cifar100
import eval_ood

################################################################################
# Model Definition 
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1)         #first convolutional layer, input is 3 because the image is RGB, output is 64 feature maps
        self.batch_normal_1 = nn.BatchNorm2d(num_features = 64)                                                 #normalize the outputs of the first convolutional layer, assists in training stability

        self.conv_layer_2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)        #second convolutional layer, input is 64 from the previous layer and output is 128
        self.batch_normal_2 = nn.BatchNorm2d(num_features = 128)                                                 #normalize the outputs of the second convolutional layer

        self.conv_layer_3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)       #third convolutional layer, input is 128 from the previous layer and output is 256
        self.batch_normal_3 = nn.BatchNorm2d(num_features = 256)                                                 #normalize the outputs of the third convolutional layer

        self.conv_layer_4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)      #fourth convolutional layer, input is 256 from the previous layer and output is 516
        self.batch_normal_4 = nn.BatchNorm2d(num_features = 512)                                                #normalize the outputs of the fourth convolutional layer

        self.pooling_layer = nn.MaxPool2d(kernel_size = 2, stride = 2)                                          #pooling layer is used to reduce the spatial size in half
        
        self.fully_connected_1 = nn.Linear(in_features = 512 * 2 * 2, out_features = 1024)                      #first fully connected layer, takes input as the flattened final convolutional layer output
        self.fully_connected_2 = nn.Linear(in_features = 1024, out_features = 100)                              #second fully connected layer, output is set to 100 classes for the CIFAR-100 

        self.dropout_layer = nn.Dropout(p = 0.5)                                                                #dropout layer is added to prevent overfitting, has 50% probability of randomly changing input to a 0
    
    def forward(self, x):
        x = F.relu(self.batch_normal_1(self.conv_layer_1(x)))        #first convolutional layer -> batch normalization -> activation -> pooling
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)        

        x = F.relu(self.batch_normal_2(self.conv_layer_2(x)))        #second convolutional layer -> batch normalization -> activation -> pooling
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)        

        x = F.relu(self.batch_normal_3(self.conv_layer_3(x)))       #third convolutional layer -> batch normalization -> activation -> pooling
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)        

        x = F.relu(self.batch_normal_4(self.conv_layer_4(x)))       #fourth convolutional layer -> batch normalization -> activation -> pooling
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)        

        x = x.view(-1, 512 * 2 * 2)                                 #flatten the output

        x = F.relu(self.fully_connected_1(x))       
        x = self.dropout_layer(x)      

        x = self.fully_connected_2(x)      
        return x

################################################################################
# Training Function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()                   #clear/initialize the gradients from the last step
        outputs = model(inputs)                 #forward pass
        loss = criterion(outputs, labels)       #loss function calculation
        loss.backward()                         #backpropagate
        optimizer.step()                        #update model

        running_loss += loss.item()             #calculate cumulative loss
        _, predicted = outputs.max(1)           #choose the class that has the highest score

        total += labels.size(0)                 #total labels seen 
        correct += predicted.eq(labels).sum().item()    #how many of the total labels seen were correct

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total     
    return train_loss, train_acc               #average loss and accuracy per epoch


################################################################################
# Validation Function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)                 #forward pass
            loss = criterion(outputs, labels)       #loss function

            running_loss += loss.item()             #accumulat loss values
            _, predicted = outputs.max(1)           #class with highest score 

            total += labels.size(0)                 #total number samples
            correct += predicted.eq(labels).sum().item()    #number of correct predictions

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():

    CONFIG = {
        "model": "Model_1",   
        "batch_size": 32, 
        "learning_rate": 0.001,
        "epochs": 50,  #Using the best performing epoch value from model_1_epoch_test
        "num_workers": 4, 
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data", 
        "ood_dir": "./data/ood-test",
        "seed": 42,
    }

    ############################################################################
    #      Data Transformation
    ############################################################################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                           #randomly crop 32x32 patches from the images with a 4-pixel padding, from ChatGPT       
        transforms.RandomHorizontalFlip(),                                              #randomly flip images horizontally, from ChatGPT
        transforms.ToTensor(),                                                          #convert train images to Pytorch tensors
        transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))         #normalize, using CIFAR-100 mean and standard deviation, values were provided by ChatGPT
        ])

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),                                                          #convert train images to Pytorch tensors
        transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))         #normalize, using CIFAR-100 mean and standard deviation, values were provided by ChatGPT
    ])

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))                                               #80% data for training
    val_size = len(trainset) - train_size                                               #20% data for testing
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])      

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = CONFIG['batch_size'], shuffle = True, num_workers = CONFIG['num_workers'])         #dataloader for training dataset
    valloader = torch.utils.data.DataLoader(valset, batch_size = CONFIG['batch_size'], shuffle = False, num_workers = CONFIG['num_workers'])            #dataloader for validation dataset

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)                                        #load CIFAR-100 test dataset and tranform 
    testloader = torch.utils.data.DataLoader(testset, batch_size = CONFIG['batch_size'], shuffle = False, num_workers = CONFIG['num_workers'])          #dataloader for test dataset
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = SimpleCNN()     
    model = model.to(CONFIG["device"])   

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()                                                           #cross-entropy loss
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum = 0.9)       #SGD optimizer with momentum -> meant to improve convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)               #reduce learning rate by half every 10 epochs

    ############################################################################
    # Training Loop 
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    model.load_state_dict(torch.load("best_model.pth"))     #added to ensure that being evaluated on a fresh set, used chat for this

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood_model_1.csv", index=False)
    print("submission_ood_model_1.csv created successfully.")

if __name__ == '__main__':
    main()
