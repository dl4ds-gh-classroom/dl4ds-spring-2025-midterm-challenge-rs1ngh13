import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)     #comment
        self.conv_layer_2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)        #comment
        self.conv_layer_3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)       #comment
        self.fully_connected_1 = nn.Linear(in_features = 128 * 4 * 4, out_features = 256)       #comment
        self.fully_connected_2 = nn.Linear(in_features = 256, out_features = 100)       #comment
        self.dropout_layer = nn.Dropout(p = 0.5)        #comment
    
    def forward(self, x):
        x = F.relu(self.conv_layer_1(x))        #comment
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)        #comment
        x = F.relu(self.conv_layer_2(x))        #comment
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)        #comment
        x = F.relu(self.conv_layer_3d(x))       #comment
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)        #comment
        x = x.view(-1, 128 * 4 * 4)     #comment
        x = F.relu(self.fully_connected_1(x))       #comment
        x = self.droupout_layer(x)      #comment
        x = self.fully_connected_2(x)       #comment
        return x

################################################################################
# Define a one epoch training function
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

        optimizer.zero_grad()       #comment
        outputs = model(inputs)     #comment
        loss = criterion(outputs, labels)       #comments
        loss.backward()         #comment
        optimizer.step()        #comment

        running_loss += loss.item()     #comment
        _, predicted = outputs.max(1)       #comment

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
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

            outputs = model(inputs)     #comment
            loss = criterion(outputs, labels)       #comment

            running_loss += loss.item()     #comment
            _, predicted = output.max(1)        #comment 

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": "MyModel",   # Change name when using a different model
        "batch_size": 8, # run batch size finder to find optimal batch size
        "learning_rate": 0.1,
        "epochs": 5,  # Train for longer in a real scenario
        "num_workers": 4, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),      #comment
        transforms.RandomHorizontalFlip(),       #comment
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441),(0.268, 0.257, 0.276))       #looked up and used values of CIFAR-100 mean and std
        ])
        
    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transform.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441),(0.268, 0.257, 0.276))       #looked up and used values of CIFAR-100 mean and std
    ])      #was not augmented, simply normalized

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = ...   ### TODO -- Calculate training set size
    val_size = ...     ### TODO -- Calculate validation set size
    trainset, valset = ...  ### TODO -- split into training and validation sets

    ### TODO -- define loaders and test set
    trainloader = ...
    valloader = ...

    # ... (Create validation and test loaders)
    testset = ...
    testloader = ...
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = ...   # instantiate your model ### TODO
    model = model.to(CONFIG["device"])   # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = ...   ### TODO -- define loss criterion
    optimizer = ...   ### TODO -- define optimizer
    scheduler = ...  # Add a scheduler   ### TODO -- you can optionally add a LR scheduler


    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Save to wandb as well

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()
