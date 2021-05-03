from modelbuilder import get_default_model
import torch
from dataloader import get_loader, MonochromeDataset as TrainDataset
import torch.nn as nn
from torch.autograd import Variable
import time
import json
from utils import haar_wavelet, extract_borders
from metrics import *

pretrained_path = "/home/dvision/Desktop/Thomas/Networks/pretrained/segnet/border_loss_epoch_99.pth"
model_classes = 2
img_classes = 2
in_channels = 1
epochs = 100
batch_size = 6
img_size = 512
mask_size = 512
storage_freq = 1
train_path = "/home/thomas/Pictures/unzipped/modified/traingdata/augmented"
test_path = "/home/thomas/Pictures/unzipped/modified/testdata"
learn_rate = 1e-4

metric = DistanceThetaMetric(1)
border_metric = BorderMetric(extract_borders, 1, metric)

train_set = TrainDataset(train_path, img_size=img_size, augmentations=False, class_channels=img_classes, mask_size=mask_size)
train_set.add_data("img", "mask")

if test_path is None:
    train_set, test_set = train_set.split_dset(0.2)
else:
    test_set = TrainDataset(test_path, img_size=img_size, class_channels=img_classes)
    test_set.add_data("img", "mask")

train_loader = get_loader(train_set, batch_size=batch_size)
test_loader = get_loader(test_set, 1)

model = get_default_model(in_channels, model_classes).cuda()

if pretrained_path is not None:
    model.load_state_dict(torch.load(pretrained_path))

optimizer = torch.optim.Adam(model.parameters(), learn_rate)

def run_dataset(model, dset, epoch, train):
    if train:
        model.train()
    else:
        model.eval()

    batch_count = len(train_loader)
    min_loss = torch.tensor(1000000000000.).cuda()
    mean_loss = torch.tensor(0.).cuda()
    min_metr = torch.tensor(1.)
    mean_metr = torch.tensor(0.)
    max_metr = torch.tensor(0.)

    for i, pack in enumerate(train_loader):
        imgs, masks, names = pack

        imgs = Variable(imgs).cuda()
        masks = Variable(masks).cuda()

        preds = model(imgs)

        masks = torch.cat([masks, extract_borders(masks[:,0].unsqueeze(1))], 1)
        preds = torch.cat([preds, extract_borders(preds[:,0].unsqueeze(1))], 1)

        metr = metric.dice(preds, masks) * border_metric.dice(preds, masks)

        mean_metr += metr
        if metr < min_metr:
            min_metr = metr
        if metr > max_metr:
            max_metr = metr

        loss = nn.functional.binary_cross_entropy(preds, masks)

        mean_loss += loss
        if loss < min_loss:
            min_loss = loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not i % 20:
            print("epoch:", epoch, "batch:", i, "/", batch_count, "loss:", loss.data.detach().cpu().numpy().item(), "metric:", metr.numpy().item())

    mean_loss /= batch_count
    mean_metr /= batch_count

    min_loss = min_loss.detach().cpu().numpy().item()
    mean_loss = mean_loss.detach().cpu().numpy().item()

    min_metr = min_metr.numpy().item()
    mean_metr = mean_metr.numpy().item()
    max_metr = max_metr.numpy().item()

    print("min loss:", min_loss, "mean loss:", mean_loss, "min metr:", min_metr, "mean metr:", mean_metr, "max metr:", max_metr)
    return min_loss, mean_loss, min_metr, mean_metr, max_metr

loss_dict = {
    "duration": [],
    "min_loss": [],
    "mean_loss": [],
    "min_metr": [],
    "mean_metr": [],
    "max_metr": [],

    "val_min_loss": [],
    "val_mean_loss": []
}

for epoch in range(epochs):
    timestamp = time.time()

    min_loss, mean_loss, min_metr, mean_metr, max_metr = run_dataset(model, train_loader, epoch, True)
    loss_dict["duration"].append(time.time() - timestamp)

    loss_dict["min_loss"].append(min_loss)
    loss_dict["mean_loss"].append(mean_loss)
    loss_dict["min_metr"].append(min_metr)
    loss_dict["mean_metr"].append(mean_metr)
    loss_dict["max_metr"].append(max_metr)

    if not epoch % storage_freq:
        torch.save(model.state_dict(), "outputs/epoch_" + str(epoch) + '.pth' )
        
    print("epoch", epoch, "needed", loss_dict["duration"][-1], "s to finish")

    if not epoch % 20:
        print("evaluating model")
        val_min_loss, val_mean_loss = run_dataset(model, test_loader, epoch, False)
        loss_dict["val_min_loss"].append(val_min_loss.cpu().detach().numpy().item())
        loss_dict["val_mean_loss"].append(val_mean_loss.cpu().detach().numpy().item())

    with open("outputs/history.json", "w") as history:
        history.write(json.dumps(loss_dict))