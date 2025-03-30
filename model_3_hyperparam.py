import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from tqdm.auto import tqdm  # For progress bars
from itertools import product
import matplotlib.pyplot as plt
import eval_cifar100
import eval_ood
from itertools import product


################################################################################
# Model Definition
################################################################################
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()        
        self.model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)                     #Load ResNet-18 with pretrained weights
        self.model.fc = nn.Linear(in_features = self.model.fc.in_features, out_features = 100)      #change the final layer to work with CIFAR-100 

    def forward(self, x):
        return self.model(x)                                                                        #forward pass calls ResNet model

################################################################################
# Train Function
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

        optimizer.zero_grad()                       #clear/initialize the gradients from the last step
        outputs = model(inputs)                     #forward pass
        loss = criterion(outputs, labels)           #loss function calculation
        loss.backward()                             #backpropagate
        optimizer.step()                            #update model

        running_loss += loss.item()                 #calculate cumulative loss
        _, predicted = outputs.max(1)               #choose the class that has the highest score

        total += labels.size(0)                     #total labels seen 
        correct += predicted.eq(labels).sum().item()    #how many of the total labels seen were correct

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


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

            running_loss += loss.item()             #accumulate loss values
            _, predicted = outputs.max(1)           #class with highest score

            total += labels.size(0)                 #total number samples
            correct += predicted.eq(labels).sum().item()    #number of correct predictions

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():

    batch_sizes = [32, 64]                          #list of batch sizes to test
    learning_rates = [0.01, 0.001, 0.0001]          #list of learning rates to test
    step_sizes = [10, 20]                           #list of step_sizes to test
    gammas = [0.1, 0.5]                             #list of gammas to test
    optimizers = ['SGD', 'ADAM']                    #list of optimizers to test
    results = []                                    #initialize list to store validation accuracy for each tested epoch
    best_val_acc = 0.0                              #initialize variable to track highest validation 
    best_config = None                              #initialize configuration to store best model param

    for batch, learning, opt, step_size, gamma in product(batch_sizes, learning_rates, optimizers, step_sizes, gammas):             #loop over all combinations of the parameters
        print(f"\n Training with batch = {batch}, learning = {learning}, optim = {opt}, step_size = {step_size}, gamma = {gamma} ")




        CONFIG = {
            "model": "Model_3",   
            "batch_size": batch, 
            "learning_rate": learning,
            "optimizer": opt,
            "step_size": step_size,
            "gamma": gamma, 
            "epochs": 25,               #used only 25 epochs to test parameters, increased to optimal epoch value 
            "num_workers": 4, 
            "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
            "data_dir": "./data",  
            "ood_dir": "./data/ood-test",
            "seed": 42,
        }

        ############################################################################
        #      Data Transformation (Example - You might want to modify) 
        ############################################################################

        transform_train = transforms.Compose([
            transforms.ToTensor(),                                                          #convert train images to Pytorch tensors
            transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))         #normalize, using CIFAR-100 mean and standard deviation, values were provided by ChatGPT
            ])

        # Validation and test transforms (NO augmentation)
        transform_test = transforms.Compose([
            transforms.ToTensor(),                                                          #convert test images to Pytorch tensors
            transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))         #noramlize, using CIFAR-100 mean and standard deviation, values were provided by ChatGPT  
        ])

        ############################################################################
        #       Data Loading
        ############################################################################

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform_train)

        # Split train into train and validation (80/20 split)
        train_size = int(0.8 * len(trainset))                                                   #80% data for training
        val_size = len(trainset) - train_size                                                   #20% data for testing
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])      

        trainloader = torch.utils.data.DataLoader(trainset, batch_size = CONFIG['batch_size'], shuffle = True, num_workers = CONFIG['num_workers'])     #dataloader for training dataset
        valloader = torch.utils.data.DataLoader(valset, batch_size = CONFIG['batch_size'], shuffle = False, num_workers = CONFIG['num_workers'])        #dataloader for validation dataset

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)                                    #load CIFAR-100 test dataset and transorm
        testloader = torch.utils.data.DataLoader(testset, batch_size = CONFIG['batch_size'], shuffle = False, num_workers = CONFIG['num_workers'])      #dataloader for test dataset
    
        ############################################################################
        #   Instantiate model and move to target device
        ############################################################################
        model = ResNet18()     
        model = model.to(CONFIG["device"]) 

        ############################################################################
        # Loss Function, Optimizer and optional learning rate scheduler
        ############################################################################
        criterion = nn.CrossEntropyLoss()                                                               #cross-entropy loss

        if opt == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)         #test SGD optimizer
        elif opt == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr = CONFIG['learning_rate'])                    #test ADAM optimizer
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = CONFIG["step_size"], gamma = CONFIG["gamma"])  #test step_size and gamma

        for epoch in range(CONFIG["epochs"]):
            train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
            val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
            scheduler.step()

        print(f"Training Complete with val_acc = {val_acc:.3f}%")
        results.append({"learning": learning, "batch": batch, "optim": opt, "step_size": step_size, "gamma": gamma, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = {"learning": learning, "batch": batch, "optim": opt, "step_size": step_size, "gamma": gamma}
            torch.save(model.state_dict(), "best_model.pth")

    print("\n Best Model Config:")
    print(best_config)
    print(f"val_acc = {best_val_acc:.3f}%")

    final_model = ResNet18().to(CONFIG["device"])
    final_model.load_state_dict(torch.load("best_model.pth"))

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(final_model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood_model_3.csv", index=False)
    print("submission_ood_model_3.csv created successfully.")

if __name__ == '__main__':
    main()