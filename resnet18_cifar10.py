# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #
import os, sys

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

import torchvision.transforms as transforms
import csv
import optim_qnvb as qnvb 

import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--optimizer', type=str, default='qnvb', help='Specify optimizer: {sgdm, adam, qnvb}')

# Arguments for SGVB and QNVB
parser.add_argument('--num_eval', type=int, default=4, help='Number of quadrature evaluations')
parser.add_argument('--quadrature', default='hadamard_cross', help='Quadrature type: {hadamard_cross, mc, qmc1, qmc2}')
parser.add_argument('--sigma_min', type=float, default=1e-3, help='Minimum and initial standard deviation')
parser.add_argument('--sigma_max', type=float, default=5e-2, help='Maximum standard deviation')
parser.add_argument('--sigma_tag', type=str, default='./sigma', help='Path and prefix for sigma csv.')

# Arguments for SGDM, Adam, and SGVB
parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--learn_reduce', type=float, default=1.05, help='Factor to reduce learning rate per epoch.')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 coefficient.')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 coefficient.')
parser.add_argument('--eps', type=float, default=1E-8, help='Adam eps coefficient.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient for SGDM.')

# Arguments for QNVB
parser.add_argument('--likelihood_weight', type=float, default=40000., help='Weight of training dataset')
parser.add_argument('--likelihood_increase_factor', type=float, default=1.05, help='Factor to increase weight each epoch.')
parser.add_argument('--scale_min', type=float, default=0.99, help='Minimum scaling factor per step')
parser.add_argument('--scale_max', type=float, default=1.01, help='Maximum scaling factor per step')

# Arguments for all
parser.add_argument('--num_epochs', type=int, default=80, help='Total number of training epochs')
parser.add_argument('--num_valid_data', type=int, default=10000, help='Number of validation data')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size of the train set')
parser.add_argument('--dataset_root', type=str, default='./data/CIFAR10', help='Path for CIFAR10 data.')
parser.add_argument('--seed', type=int, default=0, help='random seed for the run.')
parser.add_argument('--log_dir', type=str, default='./log', help='Path for tensorboard log and file prefix for csv data.')
parser.add_argument('--train_tag', type=str, default='./train', help='Path and prefix for training csv.')
args = parser.parse_args()

writer = SummaryWriter(log_dir=args.log_dir)
writer.add_text('Input Arguments:', str(args), 0)

traincsv = "{}{:02d}.csv".format(args.train_tag, args.seed)

# Reproducibility
torch.manual_seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True, transform=transform, download=True)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000 - args.num_valid_data, args.num_valid_data])
test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

model = models.resnet18(weights=None).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduction='mean')
if args.optimizer == 'qnvb':
    optimizer = qnvb.Qnvb(model.parameters(), device=device,
                      num_eval=args.num_eval, quadrature=args.quadrature,
                      sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                      scale_min=args.scale_min, scale_max=args.scale_max,
                      eps=args.eps, lr=args.learn_rate,
                      likelihood_weight=args.likelihood_weight)
    print(optimizer.initial_state_str())
    writer.add_text('Initial State: ', optimizer.initial_state_str(), 0)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
elif args.optimizer == 'sgdm':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learn_rate, momentum=args.momentum)

# For updating learning rate in Adam and SGVB
def update_lr(optimizer, lr):
    for grp in optimizer.param_groups:
        grp['lr'] = lr

# Train the model
report_steps = 100
total_step = len(train_loader)
csvfile = open(traincsv, "w")
csvwriter = csv.writer(csvfile)
csvwriter.writerow(['step', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'test_loss', 'test_acc'])
for epoch in range(args.num_epochs):
    train_loss = torch.tensor(0., device=device)
    valid_loss = torch.tensor(0., device=device)
    test_loss = torch.tensor(0., device=device)
    train_acc = torch.tensor(0., device=device)
    valid_acc = torch.tensor(0., device=device)
    test_acc = torch.tensor(0., device=device)
    train_count = torch.tensor(0, device=device)
    valid_count = torch.tensor(0, device=device)
    test_count = torch.tensor(0, device=device)

    if args.optimizer != 'qnvb':
        update_lr(optimizer, args.learn_rate/(args.learn_reduce**epoch))
    if args.optimizer == 'qnvb':
        optimizer.set_likelihood_weight(args.likelihood_weight*(args.likelihood_increase_factor**epoch))

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        def model_func():
            return model(images)
        def loss_func(outputs):
            return criterion(outputs, labels)
 
        if args.optimizer == 'qnvb':
            loss, outputs = optimizer.step((model_func, loss_func))
        else:
            outputs = model_func()
            loss = loss_func(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, max_pred = torch.max(outputs, 1)
        train_loss.add_(loss*labels.size(0))
        train_acc.add_((max_pred == labels).sum())
        train_count.add_(labels.size(0))

        if (i+1) % report_steps == 0:
            cur_step = i + 1 + epoch*len(train_loader)

            # Get full validation prediction quality.
            valid_loss = torch.tensor(0., device=device)
            valid_acc = torch.tensor(0., device=device)
            valid_count = torch.tensor(0, device=device)
            for j, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                def model_func():
                    return model(images)
                def loss_func(outputs):
                    return criterion(outputs, labels)
         
                if args.optimizer == 'qnvb':
                    outputs = optimizer.evaluate_variational_predictive(model_func)
                else:
                    outputs = model_func()
        
                _, max_pred = torch.max(outputs, 1)
                valid_loss.add_(loss_func(outputs)*labels.size(0))
                valid_acc.add_((max_pred == labels).sum())
                valid_count.add_(labels.size(0))

            # Get full test prediction quality.
            test_loss = torch.tensor(0., device=device)
            test_acc = torch.tensor(0., device=device)
            test_count = torch.tensor(0, device=device)
            for j, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                def model_func():
                    return model(images)
                def loss_func(outputs):
                    return criterion(outputs, labels)
         
                if args.optimizer == 'qnvb':
                    outputs = optimizer.evaluate_variational_predictive(model_func)
                else:
                    outputs = model_func()
        
                _, max_pred = torch.max(outputs, 1)
                test_loss.add_(loss_func(outputs)*labels.size(0))
                test_acc.add_((max_pred == labels).sum())
                test_count.add_(labels.size(0))

            train_loss.div_(train_count)
            valid_loss.div_(valid_count)
            test_loss.div_(test_count)

            train_acc.div_(train_count)
            valid_acc.div_(valid_count)
            test_acc.div_(test_count)

            writer.add_scalar('train_loss', train_loss, cur_step)
            writer.add_scalar('train_acc', train_acc, cur_step)
            writer.add_scalar('valid_loss', valid_loss, cur_step)
            writer.add_scalar('valid_acc', valid_acc, cur_step)
            writer.add_scalar('test_loss', test_loss, cur_step) 
            writer.add_scalar('test_acc', test_acc, cur_step) 
            print("Epoch {}/{}".format(epoch+1, args.num_epochs), end="")
            print(" train_count: {}".format(train_count), end="")
            print(" Step: {}/{}".format(i+1, total_step), end="")
            print(" TrainLoss: {:.4f} TestLoss: {:.4f}".format(train_loss, test_loss), end="")
            print(" TrainAcc: {:.4f} TestAcc: {:.4f}".format(train_acc, test_acc), end="")
            print(".")

            # Save data as CSV.
            csvwriter.writerow([cur_step, train_loss.item(), train_acc.item(),
                                          valid_loss.item(), valid_acc.item(),
                                          test_loss.item(), test_acc.item()])

            # Reset running train loss sum
            train_loss = torch.tensor(0., device=device)
            train_acc = torch.tensor(0., device=device)
            train_count = torch.tensor(0, device=device)
 
csvfile.close()
