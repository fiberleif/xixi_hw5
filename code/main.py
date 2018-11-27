# utility modules
# from __future__ import print_function
import argparse
import os

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

# home-made modules
from models import ConvNet, FCNet
from dataset import MyDataset
import logger


def load_dataset(args, dataset_path_with_name, use_cuda):
    # load dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.dataset_name == "NIST26" or args.dataset_name == "NIST36":
        # create dataset by defining subclass of torch.utils.data.Dataset
        data_path = dataset_path_with_name[args.dataset_name]
        train_dataset = MyDataset(data_path, train=True)
        test_dataset = MyDataset(data_path, train=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        feature_size, label_size = train_dataset.get_dim()
    elif args.dataset_name == "MNIST" or args.dataset_name == "EMNIST":
        # create dataset by directly using class torch.utils.data.Dataset
        if args.dataset_name == "MNIST":
            train_dataset = datasets.MNIST('./dataset/' + args.dataset_name.lower(), train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_dataset = datasets.MNIST('./dataset/' + args.dataset_name.lower(), train=False,
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
            label_size = 10
        else:
            train_dataset = datasets.EMNIST('./dataset/' + args.dataset_name.lower(), train=True, download=True,
                                            split="balanced",
                                            transform=transforms.Compose([
                                                transforms.ToTensor()
                                            ]))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_dataset = datasets.EMNIST('./dataset/' + args.dataset_name.lower(), train=False,
                                           split="balanced",
                                           transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,**kwargs)
            label_size = 47
        feature_size = 28 * 28
    else:
        # create dataset by directly using class torchvision.datasets.ImageFolder
        data_path = dataset_path_with_name[args.dataset_name]
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        univeral_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        train_dataset = datasets.ImageFolder(data_path, transform=univeral_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_dataset = datasets.ImageFolder(data_path.replace("train", "test"), transform=univeral_transform)
        test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        if args.dataset_name == "flowers17":
            label_size = 17
        else:
            label_size = 102
        feature_size = 224 * 224 * 3
    return train_loader, test_loader, feature_size, label_size


def train(args, model, device, train_loader, optimizer, epoch):
    # train process
    model.train() # train mode of torch --- especially for bn, dropout, etc (training is not the same as testing)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, epoch):
    # test process
    model.eval() # test mode of torch --- especially for bn, dropout, etc (training is not the same as testing)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).long()
            # print(model.features(data).shape)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logger.record_tabular('epoch', epoch)
    logger.record_tabular('avg_loss', test_loss)
    logger.record_tabular('avg_accuracy', 100. * correct / len(test_loader.dataset))
    logger.dump_tabular()


def main():
    dataset_path_with_name = {'NIST26': r"./dataset/nist_data/nist26_train.mat",
                              'NIST36': r"./dataset/nist_data/nist36_train.mat",
                              'flowers17': r"./dataset/oxford-flowers17/train",
                              'flowers102': r"./dataset/oxford-flowers102/train",
                              'MNIST': None, 'EMNIST': None}
    model_architecture_with_name = {'CNN': ConvNet, 'FCN': FCNet}

    # parse arguments
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    bool_mapper = lambda str: True if 'True' in str or 'true' in str else False
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', type=bool_mapper, default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-n', '--dataset-name', choices=dataset_path_with_name.keys(), type=str, default='MNIST',
                        help= 'dataset for training (default: MNIST)')
    parser.add_argument('-m', '--model-architecture', choices=model_architecture_with_name.keys(),
                        type=str, default='CNN', help= 'model architecture (default: CNN)')
    parser.add_argument('-p', '--pretrain', type=bool_mapper, default=False, help= 'use pretrained model (default: False)')
    args = parser.parse_args()

    # set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # configure log-information
    cwd = os.path.join(os.getcwd(), 'log')
    run_name = "{0}-{1}".format(args.dataset_name, args.model_architecture) + str(args.seed)
    cwd = os.path.join(cwd, run_name)
    logger.configure(dir=cwd)

    # load data
    train_loader, test_loader, feature_size, label_size = load_dataset(args, dataset_path_with_name, use_cuda)
    logger.info('feature size: {0}'.format(feature_size))
    logger.info('label size: {0}'.format(label_size))

    # create model
    if args.pretrain:
        model = models.squeezenet1_1(pretrained=True)
        model.classifier = nn.Sequential(
            torch.nn.Linear(512 * 13 * 13, label_size)
        )
        model.forward = lambda x: F.log_softmax(model.classifier(model.features(x).view(-1, 512 * 13 * 13)), dim=1)
        optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        if args.model_architecture == "CNN":
            if "flowers" in args.dataset_name:
                channel_size = 3
                feature_size /= 3
            else:
                channel_size = 1
            model = ConvNet(feature_size, label_size, channel_size).to(device)
        else:
            model = FCNet(feature_size, label_size).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # test initial model
    test(args, model, device, train_loader, 0)

    # alternately train and test model
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, train_loader, epoch)


if __name__ == '__main__':
    main()
