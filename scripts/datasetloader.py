from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset
from torch.utils.data import random_split
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ]
)

root_dir = "/home/normansmith/blue_storage/projects/3DSegmentationLearning/data"
fulldataset_trainval = DecathlonDataset(root_dir=root_dir,task="Task09_Spleen",transform=None,download=False,seed=20,section='training')
training_dataset, validation_dataset = random_split(fulldataset_trainval,[.85,.15])
test_dataset = DecathlonDataset(root_dir=root_dir,task="Task09_Spleen",transform=None,download=False,seed=20,section='validation')

train_ds = Dataset(training_dataset,transform=train_transforms)
validation_ds = Dataset(validation_dataset,transform=val_transforms)
testing_ds = Dataset(test_dataset,transform=val_transforms)

training_dataloader = DataLoader(train_ds,shuffle=True,num_workers=0,pin_memory=True,batch_size = 8)
validation_dataloader = DataLoader(validation_ds,shuffle=False,num_workers=0,pin_memory=True,batch_size = 8)
testing_dataloader = DataLoader(testing_ds,shuffle=False,num_workers=0,pin_memory=True,batch_size = 8)

print(len(training_dataloader))
print(len(validation_dataloader))
print(len(testing_dataloader))
train_batch = next(iter(training_dataloader))
train_features = train_batch["image"]
train_labels = train_batch["label"]
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


