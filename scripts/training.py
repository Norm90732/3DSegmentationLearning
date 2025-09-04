import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#Torch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
import torchvision
#Data
from torch.utils.data import DataLoader
from datasetloader import train_ds, validation_ds, testing_ds
from torchmetrics.segmentation import DiceScore
from monai.metrics import DiceMetric, MeanIoU as MeanIoUMetric
from monai.transforms import AsDiscrete
from monai.data.utils import decollate_batch
import monai
from models.layermodules import *
from models.Basic3DUnet import *
import wandb
#Ray
import ray
from ray import tune, train
from ray.train.torch import TorchTrainer
from ray.air import ScalingConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining
from torch.amp import autocast, GradScaler

#hydra
import hydra
from omegaconf import DictConfig,OmegaConf

def createDataset(train_ds, validation_ds, testing_ds):
    return train_ds, validation_ds, testing_ds
    


def trainEpoch(model, device, dataloader,criterion,optimizer,scaler):
    if hasattr(dataloader.sampler, "set_epoch"):
        currentEpoch = 0
        dataloader.sampler.set_epoch(currentEpoch)
        
    
def validEpoch(model,device,dataloader,criterion):
        s = 1 
        return s     



def objective(config,training_dataset,validation_dataset):
    x = 1 
    #no return value for this 
    training_dataloader = DataLoader(training_dataset,
                                     shuffle=True,num_workers=0,pin_memory=True,batch_size=16)
    validation_dataloader = DataLoader(validation_dataset,
                                       shuffle=False,num_workers=0,
                                       pin_memory=True,batch_size=16)

    model = Basic3DUnet(input_channels=1,output_channel=1) # dropoutProb= Insert dropoutconfig here 
    
    model = train.torch.prepare_model(model)
    training_dataloader = train.torch.prepare_data_loader(training_dataloader)
    validation_dataloader = train.torch.prepare_data_loader(validation_dataloader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()
    
    
    
    
@hydra.main(config_path='conf',config_name="config",version_base=None)
def main(cfg:DictConfig) -> None:
     ray.init(configure_logging=False)
     training_dataset,validation_dataset,testing_dataset =createDataset(train_ds, validation_ds, testing_ds)
     
     
     train_datasetReference = ray.put(training_dataset)
     validation_datasetReference = ray.put(validation_dataset)
     testing_datasetReference = ray.put(testing_dataset)
     
     
     search_space = {
         #put config/search space here
     }
     
     #from turning tune string to live ray tune object
     for key,value in cfg.params.search_space.items(): ## change cfg here its not correct 
         search_space[key] = eval(value)
    
     
     algo = OptunaSearch(metric="validation_dice", mode='max')
     algo = ConcurrencyLimiter(algo, max_concurrent=3)
     
     
     torch_trainer = TorchTrainer(
         train_loop_per_worker=objective,
         train_loop_config=search_space
         scaling_config=ScalingConfig(num_workers=2,
                                      use_gpu=True,
                                      resources_per_worker={"CPU":5}),
     )
     
     trainableWithData = tune.with_parameters(
         torch_trainer,
         training_dataset= train_datasetReference,
         validation_dataset = validation_datasetReference
     )
     
    #Fill this in later for RunConfig using wandb
     tuner = tune.Tuner(
        trainableWithData,
        run_config=train.RunConfig,
        tune_config=tune.TuneConfig,
    )