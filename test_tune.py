import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from functools import partial
import random

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch


def seed_everything(seed=1234):
    # https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return


class MyDataset(Dataset):
    def __init__(self, x, y, device="cpu"):
        self.device = device
        self.n, self.ndim = x.shape
        self.n_classes = len(np.unique(y))
        self.x = torch.tensor(x, dtype=torch.float, device=self.device)
        self.y = torch.tensor(y, dtype=torch.long, device=self.device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n


class MLP(torch.nn.Module):
    def __init__(self, nin, nh1, nh2, nout):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(nin, nh1)
        self.fc2 = torch.nn.Linear(nh1, nh2)
        self.fc3 = torch.nn.Linear(nh2, nout)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)


def load_data():
    digits = load_digits()
    x = digits.data / digits.data.max()
    y = digits.target
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, stratify=y)
    trainset = MyDataset(x_train, y_train)
    testset = MyDataset(x_val, y_val)
    return trainset, testset


def get_data_loaders(batch_size):
    trainset, valset = load_data()
    trainloader = DataLoader(trainset, batch_size=int(batch_size), shuffle=True)
    valloader = DataLoader(valset, batch_size=int(batch_size), shuffle=True)
    return trainloader, valloader


def train(model, optimizer, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.
    running_accuracy = 0.
    model.train()
    for i, data in enumerate(dataloader, 0):
        x, y = data[0].to(device), data[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        y_pred = outputs.argmax(1).detach()
        acc = ((y_pred == y).sum() / len(y))
        running_loss += loss.item()
        running_accuracy += acc.item()
    running_loss /= (i + 1)
    running_accuracy /= (i + 1)
    # print("train loss: {:.3f}, train accuracy: {:.3f}".format(running_loss, running_accuracy))
    return running_loss, running_accuracy


def evaluate(model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.
    running_accuracy = 0.
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            x, y = data[0].to(device), data[1].to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            y_pred = outputs.argmax(1)
            acc = ((y_pred == y).sum() / len(y))
            running_loss += loss.item()
            running_accuracy += acc.item()
    running_loss /= (i + 1)
    running_accuracy /= (i + 1)
    # print("test loss: {:.3f}, test accuracy: {:.3f}".format(running_loss, running_accuracy))
    return running_loss, running_accuracy


class Trainer(tune.Trainable):
    def setup(self, config):
        # self.x = 0
        self.lr = config["lr"]
        self.nh1 = config["nh1"]
        self.nh2 = config["nh2"]
        self.batch_size = config["batch_size"]

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.train_loader, self.val_loader = get_data_loaders(self.batch_size)
        nin = self.train_loader.dataset.ndim
        nout = self.train_loader.dataset.n_classes
        self.model = MLP(nin, self.nh1, self.nh2, nout)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return
    
    def step(self):
        train(self.model, self.optimizer, self.train_loader, self.device)
        loss, acc = evaluate(self.model, self.val_loader, self.device)
        # self.x += 1
        return {"loss": loss, "accuracy": acc}
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return


def train_model(config, epochs):
    train_loader, val_loader = get_data_loaders(config["batch_size"])
    nin = train_loader.dataset.ndim
    nout = train_loader.dataset.n_classes
    model = MLP(nin, config["nh1"], config["nh2"], nout)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    for epoch in range(epochs):
        train(model, optimizer, train_loader, torch.device("cpu"))
        loss, acc = evaluate(model, val_loader, torch.device("cpu"))
        tune.report(loss=loss, accuracy=acc)
    return


def set_gpu_id(ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in ids])
    return


seed_everything()

parser = argparse.ArgumentParser(description="test_tune")
parser.add_argument("--gpu", action="store_true", help="use gpu (default False)")
parser.add_argument("--oop", action="store_true", help="oop setting (default False)")
parser.add_argument("--num_samples", type=int, default=2, help="number of samples (default 2)")
parser.add_argument("--epochs", type=int, default=2, help="number of epochs (default 2)")
args = parser.parse_args()
print(args)

epochs = args.epochs
gpus_per_trial = 0
cpus_per_trial = 2
num_samples = args.num_samples
num_cpus = 4
num_gpus = 0
local_dir = "runs"
exp_name = "test_cpu"
checkpoint_freq = int(epochs / 10)

if args.gpu:
    gpu_ids = [1, 2, 3]
    set_gpu_id(gpu_ids)
    gpus_per_trial = 1
    num_gpus = 3
    exp_name = "test_gpu"

ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

config = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "nh1": tune.choice([16, 32, 64]),
    "nh2": tune.choice([16, 32, 64]),
    "batch_size": tune.choice([2, 4, 8, 16, 32])
}

if args.oop:
    analysis = tune.run(
        Trainer,
        local_dir=local_dir,
        name=exp_name,
        stop={"training_iteration": epochs},
        config=config,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        num_samples=num_samples,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        max_failures=5,
        search_alg=HyperOptSearch(metric="accuracy", mode="max")
    )
else:
    analysis = tune.run(
        partial(train_model, epochs=epochs),
        local_dir="checkpoints",
        name=exp_name,
        stop={"training_iteration": epochs},
        config=config,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        num_samples=num_samples,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        max_failures=5,
        search_alg=HyperOptSearch(metric="accuracy", mode="max")
    )

print("best config: ", analysis.get_best_config(metric="accuracy", mode="max"))

df = analysis.dataframe()
df.to_csv("analysis.csv")
