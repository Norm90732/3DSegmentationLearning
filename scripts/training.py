import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#Torch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

#Data
from monai.data import DataLoader
from datasetloader import train_ds, validation_ds, testing_ds
from torchmetrics.segmentation import DiceScore
from monai.metrics import DiceMetric, MeanIoU as MeanIoUMetric
from monai.transforms import AsDiscrete
from monai.data.utils import decollate_batch
import monai
from models.layerModules import *
from models.Basic3DUnet import *
from models.TransUnet import * 
import wandb

#Ray
import ray
from ray import tune, train
from ray.train.torch import TorchTrainer, TorchConfig
from ray.air import ScalingConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.amp import autocast, GradScaler

#hydra
import hydra
from omegaconf import DictConfig,OmegaConf

def createDataset(train_ds, validation_ds, testing_ds):
    return train_ds, validation_ds, testing_ds
    


def trainEpoch(model, device, dataloader,criterion,optimizer,scaler,epoch,numClasses):
    model.train()
    trainDiceMetric = DiceMetric(include_background=False,reduction='mean')
    postPrediction = AsDiscrete(argmax=True,to_onehot=numClasses) 
    postMask = AsDiscrete(to_onehot=numClasses) 
    running_loss = 0.0 
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    
    
    for batch_data in dataloader:
        images, masks = batch_data["image"].to(device), batch_data["label"].to(device).long()
        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs,masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        pred_onehot = [postPrediction(i) for i in decollate_batch(outputs)]
        mask_onehot = [postMask(i) for i in decollate_batch(masks.unsqueeze(1))]
        
        trainDiceMetric(y_pred=pred_onehot,y=mask_onehot)
    
    trainingOverallDice = trainDiceMetric.aggregate().item()
    trainDiceMetric.reset()
    trainingLoss = running_loss/len(dataloader)
    return trainingOverallDice,trainingLoss
        
    
def validEpoch(model,device,dataloader,criterion,numClasses):
        model.eval()
        valDiceMetric = DiceMetric(include_background=False,reduction='mean')
        postPrediction = AsDiscrete(argmax=True,to_onehot=numClasses) #add config element here
        postMask = AsDiscrete(to_onehot=numClasses) #add config element here
        runningValLoss = 0.0
        
        with torch.no_grad():
            for batch_data in dataloader:
                images, masks = batch_data["image"].to(device), batch_data["label"].to(device).long()
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs,masks)
                pred_onehot = [postPrediction(i) for i in decollate_batch(outputs)]
                mask_onehot = [postMask(i) for i in decollate_batch(masks.unsqueeze(1))]
                valDiceMetric(y_pred=pred_onehot,y=mask_onehot)
                runningValLoss += loss.item()
        valOveralldice = valDiceMetric.aggregate().item()
        valDiceMetric.reset()
        epochValLoss = runningValLoss/len(dataloader)
        return valOveralldice,epochValLoss



def objective(config):
    
    training_dataset_ref = config.pop("training_dataset")
    validation_dataset_ref = config.pop("validation_dataset")
    
    training_dataset = ray.get(training_dataset_ref)
    validation_dataset = ray.get(validation_dataset_ref)
    
    numClasses = config["num_classes"]
    batch_size = config["batch_size"]
    numEpochs = config["numEpochs"]
    
    training_dataloader = DataLoader(training_dataset,
                                     shuffle=True,num_workers=4,pin_memory=True,batch_size=batch_size)
    
    validation_dataloader = DataLoader(validation_dataset,
                                       shuffle=False,num_workers=4,
                                       pin_memory=True,batch_size=batch_size)
    
    if config["name"] == "basicUnet":
        model = Basic3DUnet(input_channels=config['input_channels'],
                            output_channel=numClasses,
                            dropoutProb=config['dropout']) 
    elif config["name"] == "TransUnet":
        model = TransUnet(input_channels=config['input_channels'],
                          heightImg=config['heightImg'],
                          widthImg=config['widthImg'],
                          depthImg=config['depthImg'],
                          patchsize=config['patchsize'],
                          output_channel=numClasses,
                          dropoutProb=config['dropoutProb'],
                          embedDim=config['embedDim'],
                          ffDim=config['ffDim'],
                          numHeads=config['numHeads'],
                          dropoutMLP=config['dropoutMLP'],
                          dropoutAttention=config['dropoutAttention'],
                          numEncoderBlock=config['numEncoderBlock'],
                          )
    
    lambda_crossEntropy_val = 1- config["lambda_dice"]
    
    criterion = monai.losses.DiceCELoss(
        include_background=False, softmax = True, to_onehot_y=True,
        lambda_dice = config['lambda_dice'], lambda_ce = lambda_crossEntropy_val
    )
    
    optimizer = optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config["T_0"], T_mult=config['T_mult'], eta_min = config['eta_min']
    )
    model = train.torch.prepare_model(
    model,
    parallel_strategy_kwargs={"find_unused_parameters": True}
    )
    training_dataloader = train.torch.prepare_data_loader(training_dataloader)
    validation_dataloader = train.torch.prepare_data_loader(validation_dataloader)

    
    
    
    device = train.torch.get_device()
    model.to(device)

    scaler = GradScaler()
    
    for epoch in range(numEpochs): #pass from cfg
        trainingDice, trainingLoss = trainEpoch(model,device,training_dataloader,criterion,optimizer,scaler,epoch,numClasses=numClasses)
        validationDice, validationLoss = validEpoch(model,device, validation_dataloader,criterion,numClasses=numClasses)
        scheduler.step()
    
        train.report({
            "trainingLoss": trainingLoss,
            "trainingDice": trainingDice,
            "validationLoss": validationLoss,
            "validationDice": validationDice,
            "epoch": epoch
        })
    
def test(config):
    training_dataset_ref = config.pop("training_dataset")
    validation_dataset_ref = config.pop("validation_dataset")
    test_dataset_ref = config.pop("testing_dataset")
    
    training_dataset = ray.get(training_dataset_ref)
    validation_dataset = ray.get(validation_dataset_ref)
    testing_dataset = ray.get(test_dataset_ref)
    
    numClasses = config['num_classes']
    batch_size = config["batch_size"]
    numTestEpochs = config["numEpochsTest"]
    
    
    training_dataloader = DataLoader(training_dataset,
                                     shuffle=True,num_workers=4,pin_memory=True,batch_size=batch_size)
    
    validation_dataloader = DataLoader(validation_dataset,
                                       shuffle=False,num_workers=4,
                                       pin_memory=True,batch_size=batch_size)
    testing_dataloader = DataLoader(testing_dataset,num_workers=4,
                                       pin_memory=True,batch_size=batch_size)
    if config["name"] == "basicUnet":
        model = Basic3DUnet(input_channels=config['input_channels'],
                            output_channel=numClasses,
                            dropoutProb=config['dropout']) 
    elif config["name"] == "TransUnet":
        model = TransUnet(input_channels=config['input_channels'],
                          heightImg=config['heightImg'],
                          widthImg=config['widthImg'],
                          depthImg=config['depthImg'],
                          patchsize=config['patchsize'],
                          output_channel=numClasses,
                          dropoutProb=config['dropoutProb'],
                          embedDim=config['embedDim'],
                          ffDim=config['ffDim'],
                          numHeads=config['numHeads'],
                          dropoutMLP=config['dropoutMLP'],
                          dropoutAttention=config['dropoutAttention'],
                          numEncoderBlock=config['numEncoderBlock'],
                          )
        
    lambda_crossEntropy_val = 1- config["lambda_dice"]
    
    criterion = monai.losses.DiceCELoss(
        include_background=False, softmax = True, to_onehot_y=True,
        lambda_dice = config['lambda_dice'], lambda_ce = lambda_crossEntropy_val
    )
    
    optimizer = optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config["T_0"], T_mult=config['T_mult'], eta_min = config['eta_min']
    )
    model = train.torch.prepare_model(
    model,
    parallel_strategy_kwargs={"find_unused_parameters": True}
    )
    training_dataloader = train.torch.prepare_data_loader(training_dataloader)
    validation_dataloader = train.torch.prepare_data_loader(validation_dataloader)
    testing_dataloader = train.torch.prepare_data_loader(testing_dataloader)
    
    device = train.torch.get_device()
    model.to(device)
    name = config['name']
    scaler = GradScaler()
    bestValidationDice = -1.0
    best_model_state = None
    
    testDiceMetric = DiceMetric(include_background=False,reduction='mean')
    postPrediction = AsDiscrete(argmax=True,to_onehot=numClasses) #add config element here
    postMask = AsDiscrete(to_onehot=numClasses) #add config element here
    for epoch in range(numTestEpochs):#Pass from cfg
        trainingDice, trainingLoss = trainEpoch(model,device,training_dataloader,criterion,optimizer,scaler,epoch,numClasses=numClasses)
        validationDice, validationLoss = validEpoch(model,device, validation_dataloader,criterion,numClasses=numClasses)
        currentValidationDice = validationDice
        if currentValidationDice > bestValidationDice:
            bestValidationDice = currentValidationDice
            best_model_state = model.state_dict().copy()
        scheduler.step()
        train.report({
            "final_run_training_loss": trainingLoss,
            "final_run_training_dice": trainingDice,
            "final_run_validation_loss": validationLoss,
            "final_run_validation_dice": validationDice,
            "epoch": epoch  
        })
    if train.get_context().get_world_rank() == 0:
        print("\n--- DEBUG: Entering model saving block ---")
        storage_path = "/blue/uf-dsi/normansmith/projects/3DSegmentationLearning"
    
        model_save_dir = os.path.join(storage_path, "final_models")
        print(f"Model save directory is: {model_save_dir}")
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, f"{name}.pth")
        print(f"Attempting to save model to: {save_path}")
        torch.save(best_model_state, save_path)
    model.load_state_dict(best_model_state)
    model.eval()
        
    with torch.no_grad():
        for batch_data in testing_dataloader:
            images, masks = batch_data["image"].to(device), batch_data["label"].to(device).long()
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
            pred_onehot = [postPrediction(i) for i in decollate_batch(outputs)]
            mask_onehot = [postMask(i) for i in decollate_batch(masks.unsqueeze(1))]
            testDiceMetric(y_pred=pred_onehot,y=mask_onehot)

    testOverallDice = testDiceMetric.aggregate().item()
    testDiceMetric.reset()
    train.report({"final_test_dice": testOverallDice})
    return testOverallDice

        
        
        
        
        
    
@hydra.main(config_path='../configs',config_name="config",version_base=None)
def main(cfg:DictConfig) -> None:
     ray.init(configure_logging=False)
     
     training_dataset,validation_dataset,testing_dataset =createDataset(train_ds, validation_ds, testing_ds)
     
     
     train_datasetReference = ray.put(training_dataset)
     validation_datasetReference = ray.put(validation_dataset)
     testing_datasetReference = ray.put(testing_dataset)
     
     hyperparam_search = {key: eval(value) for key, value in cfg.params.search_space.items()}
     model_search_space = {key: eval(value) for key, value in cfg.model.search_space.items()}
     hyperparam_search.update(model_search_space)
     
     param_space = {
        "train_loop_config": hyperparam_search
     }
    
     base_config = {}
     base_config["training_dataset"] = train_datasetReference
     base_config["validation_dataset"] = validation_datasetReference
     base_config["testing_dataset"] = testing_datasetReference
     base_config.update(cfg.training)
     base_config.update(cfg.testing)
     model_static_params = {k: v for k, v in cfg.model.items() if k != 'search_space'}
     base_config.update(model_static_params)
    
     
     algo = OptunaSearch(metric="validationDice", mode='max')
     algo = ConcurrencyLimiter(algo, max_concurrent=cfg.ray_config.max_concurrent_trials)
     
     torch_config = TorchConfig()

     torch_trainer = TorchTrainer(
         train_loop_per_worker=objective,
         train_loop_config=base_config,
         torch_config=torch_config,
         scaling_config=ScalingConfig(num_workers=cfg.ray_config.num_workers,
                                      use_gpu=True,
                                      resources_per_worker={"CPU":cfg.ray_config.cpus_per_worker}),
     )
     
     scheduler = AsyncHyperBandScheduler(
         metric='validationDice',
         mode = "max",
         grace_period=5,
         reduction_factor=2,
    )
    
     tuner = tune.Tuner(
        torch_trainer,
        param_space=param_space,
        run_config=train.RunConfig(
            name="HyperParamSweepUnet",callbacks=[WandbLoggerCallback(project=cfg.wandb.project,log_config=cfg.wandb.log_config)]
            ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=cfg.ray_config.num_samples),
    ) 
     
     results = tuner.fit()
     best_result = results.get_best_result(metric="validationDice",mode="max")
     
     best_hyperparameters_from_sweep = best_result.config['train_loop_config']
     final_config = base_config.copy()
     final_config.update(best_hyperparameters_from_sweep)
     
     final_wandb_callback = WandbLoggerCallback(
        project=cfg.wandb.project,
        name="final-model-test-run",
        config=final_config
    )
     final_trainer = TorchTrainer(
        train_loop_per_worker=test,
        train_loop_config=final_config,
        torch_config=torch_config,
        scaling_config=ScalingConfig(num_workers=cfg.ray_config.num_workers,
                                     use_gpu=True,
                                     resources_per_worker={"CPU":cfg.ray_config.cpus_per_worker}),
        
        run_config=train.RunConfig(
            name="FinalModelTraining",
            storage_path = "/blue/uf-dsi/normansmith/projects/3DSegmentationLearning",
            callbacks=[final_wandb_callback]
        ) 
      )
     final_result = final_trainer.fit()
     
     

if __name__ == "__main__":
    main()