import torch, warnings, torchvision
import typing as t
import torch.nn.functional as F
from torch.nn import Conv2d
from pathlib import Path
from datetime import datetime
from torch import nn
import numpy as np
from sklearn import metrics
from ..util import util
from collections import OrderedDict

def train_one_epoch(model, train_loader, optimizer, loss_functions, device):
    #This is the main code responsible for updating the weights of the model for a single epoch
    
    model.train() #set the model in training mode
    epoch_loss = 0
    
    for batch in train_loader: 
        imgs, targets = batch #getting imgs and target output for current batch
        
        #we have a tensor in the train_loader, move to device
        imgs = imgs.to(device= device, dtype = torch.float32)
        targets = targets.to(device= device, dtype = torch.float32)
        
        optimizer.zero_grad()  # sets to zero the gradients of the optimizer
        
        # Forward pass
        network_output = model(imgs) 
        
        #output from model is [B,C], changing to [B, C, 1]
        network_output = torch.unsqueeze(network_output, dim = -1)
        
        # Compute the loss
        loss = sum([f(network_output, targets) for f in loss_functions]) # compute the error between the network output and target output
        
        # Backward pass
        loss.backward() # compute the gradients given the loss value
        
        # update weights
        optimizer.step() # update the weights of models using the gradients and the given optimizer
        
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader.dataset)
        
    return epoch_loss

def calculate_validation_loss(model, validation_loader, loss_functions, device):
    model.eval() #set the model in evaluation mode
    val_loss = 0
    with torch.no_grad():  # disable gradient calculations during evaluation
        for batch in validation_loader: 
            imgs, targets = batch #getting imgs and target output for current batch
            
            #we have a tensor in the validation_loader, move to device
            imgs = imgs.to(device= device, dtype = torch.float32)
            targets = targets.to(device= device, dtype = torch.float32)
            
            network_output = model(imgs) #applying the model to the input images
            
            #output from model is [B,C], changing to [B, C, 1]
            network_output = torch.unsqueeze(network_output, dim = -1)            
            
            loss = sum([loss_fn(network_output, targets) for loss_fn in loss_functions]) # compute the error between the network output and target output
            
            val_loss += loss.item()
        val_loss /= len(validation_loader.dataset)
        
    return val_loss

def calculate_test_performance(model, test_loader, device, folder_output = None):
    model.eval() #set the model in evaluation mode
    
    ground_truth_labels = np.array([])
    network_labels = np.array([])
    
    #index and file_names from test images
    index = np.array(test_loader.dataset.indices)
    file_names = [test_loader.dataset.dataset.path_files[i].stem for i in index]
    
    k = 0
    with torch.no_grad():  # disable gradient calculations during evaluation
        for batch in test_loader: 
            imgs, targets = batch #getting imgs and target output for current batch
            
            #we have a tensor in the validation_loader, move to device
            imgs = imgs.to(device= device, dtype = torch.float32)
            targets = targets.to(device= device, dtype = torch.float32)
            
            network_output = model(imgs) #applying the model to the input images
            
            #targets is shape [B,C,1] changing to [B,C] and getting the class label
            targets_class = torch.argmax(torch.squeeze(targets, dim = -1), dim=1).cpu().numpy()
            
            
            #output from model is [B,C]. Calculate network labels
            network_output = torch.argmax(torch.sigmoid(network_output), dim=1).cpu().numpy()
                        
            ground_truth_labels = np.hstack((ground_truth_labels, targets_class) )
            network_labels = np.hstack((network_labels, network_output) )
            
            #saving image and prediction
            for j in range(0, len(targets_class)):
                current_output = network_output[j]
                current_target = targets_class[j]
                
                # Calculate probability map and convert to numpy array
                file_id = file_names[k]
                k+=1
                
                util.imwrite(Path(folder_output, f'{file_id}_pred_{current_output}_target_{current_target}.tif'), (255*imgs[j,:,:,:]).cpu().numpy().astype(np.uint8))
    
    for i in range(0, len(file_names)):
        print(f'{file_names[i]} --- (gt, predicted) = ({ground_truth_labels[i]}, {network_labels[i]})')
            
    confusion_matrix = metrics.confusion_matrix(ground_truth_labels, network_labels)
    
    print(confusion_matrix)    
    
    
    
    return 0

def get_model_outputdir(model_output_folder):
    if not model_output_folder:
        model_output_folder = Path(Path(__file__).absolute().parent, 'model_training_results', datetime.now().strftime('%Y_%m_%d_Time_%H_%M_%S'))
        model_output_folder.mkdir(parents=True, exist_ok=True)
        if Path(__file__).parent.stem != 'core_code':
            warnings.warn(f"We assume that the parent folder of function {Path(__file__).stem} is: core_code")
    return model_output_folder

def load_model(model_path, device = 'cpu'):
    state_dict = torch.load(model_path, map_location= device)

    n_channels_input = state_dict[list(state_dict.keys())[0]].size(1)
    n_classes = state_dict[list(state_dict.keys())[-1]].size(0)
    
    model = get_model('resnet50', n_channels_input, n_classes).to(device= device)
    
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            
            name = k[10:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    
    return model

# get automatic batch size --- implementation from
# https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1
#☺made small modifications
def get_batch_size(
    device: torch.device,
    input_shape: t.Tuple[int, int, int],
    output_shape: t.Tuple[int],
    dataset_size: int,
    model_type: str = 'resnet50',
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    
    model = get_model(model_type, input_shape[0], output_shape[0])
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 1
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(*(batch_size, *output_shape), device=device)
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
      
    return batch_size

def get_model(model_type, n_channels_input, out_features):
    if model_type == 'resnet50':
        model = torchvision.models.resnet50()
        
        #Adjust the conv1 to the number of input channels
        model.conv1 = Conv2d(n_channels_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        
        #Adjust the fc to the number of output classes
        model.fc = nn.Linear(in_features=2048, 
                             out_features= out_features
                             )
        return model
    
def get_dataloader_file_names(dataset_loader, fullpath = True):
    #index and file_names from test images
    index = np.array(dataset_loader.dataset.indices)
    file_names = [dataset_loader.dataset.dataset.path_files[i] if fullpath else dataset_loader.dataset.dataset.path_files[i].stem for i in index]
    
    return file_names
    
    