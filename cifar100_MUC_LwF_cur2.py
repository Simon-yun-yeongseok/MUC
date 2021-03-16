#!/usr/bin/env python
# coding=utf-8
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
import scipy.io as sio
try:
    import cPickle as pickle
except:
    import pickle
import resnet_model_cur
import utils_pytorch
from compute_accuracy import compute_accuracy_WI
from compute_accuracy import compute_accuracy_Version1
import random

start = time.localtime()


######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--dataset_dir', default='./data/cifar-100-python', type=str)
parser.add_argument('--OOD_dir', default='./data/SVHN', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--nb_cl_fg', default=20, type=int, help='the number of classes in first group')
parser.add_argument('--nb_cl', default=20, type=int, help='Classes per group')
parser.add_argument('--nb_protos', default=0, type=int, help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default='MUC_LwF_cifar100', type=str, help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int, help='Epochs')
parser.add_argument('--val_epoch', default=10, type=int, help='Epochs')
parser.add_argument('--T', default=2, type=float, help='Temperature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, help='Beta for distialltion')
parser.add_argument('--resume', default='True', action='store_true', help='resume from checkpoint')
parser.add_argument('--random_seed', default=20, type=int, help='random seed')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--side_classifier', default=1, type=int, help='multiple classifiers')  # gmpark
parser.add_argument('--Stage3_flag', default='True', action='store_true', help='multiple classifiers')
parser.add_argument('--memory_budget', default=2000, type=int, help='Exemplars of old classes')
args = parser.parse_args()
# Set random seeds.                     # Fix the random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
print(args.random_seed)

ckp_prefix = './checkpoint/{}/MUC_LwF/step_{}_K_{}/'.format(args.dataset, args.nb_cl, args.side_classifier)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    t = t.to(device)
    return Variable(t, **kwargs)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)
train_batch_size       = 128            # Batch size for train
test_batch_size        = 100            # Batch size for test
eval_batch_size        = 100            # Batch size for eval
base_lr                = 0.1            # Initial learning rate
lr_strat               = [120, 160, 180]      # Epochs where learning rate gets decreased
lr_factor              = 0.1            # Learning rate decrease factor
custom_weight_decay    = 5e-4            # Weight Decay
custom_momentum        = 0.9            # Momentum
epochs                 = 200
val_epoch              = 10             # evaluate the model in every val_epoch
save_epoch             = 50             # save the model in every save_epoch
print(args)
Stage1_flag = True  # Train new model and new classifier
Stage3_flag = False  # Train side classifiers with Maximum Classifier Discrepancy  # gmpark
########################################
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])

trainset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=True,
                                        download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=False, transform=transform_test)
evalset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=False, transform=transform_test)

# save accuracy
top1_acc_list = np.zeros((args.nb_runs, int(args.num_classes/args.nb_cl), int(epochs/val_epoch)))
# gmpark
X_train_total = np.array(trainset.data)
Y_train_total = np.array(trainset.targets)
X_valid_total = np.array(testset.data) # test set is used as val set
Y_valid_total = np.array(testset.targets)

## Load unlabeled data from SVHN  # gmpark
# svhn_data = torchvision.datasets.SVHN(root=args.OOD_dir, download=False, transform=transform_train)
# svhn_num = svhn_data.data.shape[0]
# svhn_data_copy = svhn_data.data
# svhn_labels_copy = svhn_data.labels

# Launch the different runs
for n_run in range(args.nb_runs):
    # Select the order for the class learning
    order_name = "./checkpoint/{}_order_run_{}.pkl".format(args.dataset, n_run)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        print("Generating orders")
        order = np.arange(args.num_classes)
        np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)
    order_list = list(order)
    print(order_list)

    start_iter = 0
    for iteration in range(start_iter, int(args.num_classes/args.nb_cl)):
        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(iteration*args.nb_cl,(iteration+1)*args.nb_cl)]
        indices_train_subset = np.array([i in order[range(iteration*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_subset  = np.array([i in order[range(0,(iteration+1)*args.nb_cl)] for i in Y_valid_total])

        ## images
        X_train          = X_train_total[indices_train_subset]
        X_valid          = X_valid_total[indices_test_subset]
        ## labels
        Y_train          = Y_train_total[indices_train_subset]
        Y_valid          = Y_valid_total[indices_test_subset]

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid = np.array([order_list.index(i) for i in Y_valid])
        ############################################################
        # gmpark
        trainset.data = X_train.astype('uint8')
        trainset.targets = map_Y_train
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
        testset.data = X_valid.astype('uint8')
        testset.targets = map_Y_valid
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid), max(map_Y_valid)))
        ##############################################################
        # Add the stored exemplars to the training data
        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            indices_test_subset_ori = np.array([i in order[range(0, iteration*args.nb_cl)] for i in Y_valid_total])
            X_valid_ori = X_valid_total[indices_test_subset_ori]
            Y_valid_ori = Y_valid_total[indices_test_subset_ori]

        if iteration == start_iter:
            # base classes
            tg_model = resnet_model.resnet32(num_classes=args.nb_cl, side_classifier=args.side_classifier)  # gmpark
            tg_model = tg_model.to(device)
            ref_model = None
            num_old_classes = 0
            for param in tg_model.fc_side.parameters():
                param.requires_grad = False
        else:
            #increment classes
            ref_model = copy.deepcopy(tg_model) 
            ref_model = ref_model.to(device)
            for param in ref_model.parameters():
                param.requires_grad = False

            ## new main classifier
            num_old_classes = ref_model.fc.out_features
            in_features = ref_model.fc.in_features # dim
            new_fc = nn.Linear(in_features, args.nb_cl*(iteration+1)).cuda()
            new_fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
            new_fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
            tg_model.fc = new_fc

            ## new side classifier
            num_old_classes_side = ref_model.fc_side.out_features
            in_features = ref_model.fc.in_features # dim
            new_fc_side = nn.Linear(in_features, args.side_classifier*args.nb_cl*(iteration+1)).cuda()
            new_fc_side.weight.data[:num_old_classes_side] = ref_model.fc_side.weight.data
            new_fc_side.bias.data[:num_old_classes_side] = ref_model.fc_side.bias.data
            tg_model.fc_side = new_fc_side
            for param in tg_model.parameters():
                param.requires_grad = True

########### Stage 1: Train Multiple Classifiers for each iteration #################
        if Stage1_flag is True:
            print("Stage 1: Train the model for iteration {}".format(iteration))
            # Training
            update_params = list(tg_model.parameters())
            tg_optimizer = optim.SGD(update_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
            # tg_optimizer = optim.SGD(tg_params, lr=base_lr, weight_decay=custom_weight_decay)
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
            cls_criterion = nn.CrossEntropyLoss()
            cls_criterion.to(device)
            acc_best = 0
            for epoch in range(epochs):
                temp = 1
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    if args.cuda:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                    if iteration == start_iter:
                        outputs = tg_model(inputs, side_fc=False)
                        loss_cls = cls_criterion(outputs[:, num_old_classes:(num_old_classes+args.nb_cl)], targets)
                        loss = loss_cls
                    else:
                        targets = targets - args.nb_cl * iteration
                        outputs = tg_model(inputs)
                        loss_cls = 0
                        outputs = tg_model(inputs, side_fc=False)
                        loss_cls = cls_criterion(outputs[:, num_old_classes:(num_old_classes + args.nb_cl)], targets)

                        # distillation loss for main classifier
                        old_outputs = ref_model(inputs, side_fc=False)
                        soft_target = F.softmax(old_outputs / args.T, dim=1)
                        logp = F.log_softmax(outputs[:, :num_old_classes] / args.T, dim=1)
                        loss_distill_main = -torch.mean(torch.sum(soft_target * logp, dim=1))

                        # # # distillation loss for side classifiers
                        outputs_side = tg_model(inputs, side_fc=True)
                        outputs_old_side = ref_model(inputs, side_fc=True)
                        ## discrepancy loss
                        index = args.nb_cl * args.side_classifier * (iteration-1)
                        loss_distill_side = 0
                        if args.side_classifier > 1:
                            for iter_1 in range(args.side_classifier):
                                outputs_old_side_each = outputs_old_side[:, (index + args.nb_cl * iter_1):(index + args.nb_cl * (iter_1 + 1))]
                                soft_target_side = F.softmax(outputs_old_side_each / args.T, dim=1)
                                outputs_side_each = outputs_side[:, (index + args.nb_cl * iter_1):(index + args.nb_cl * (iter_1 + 1))]
                                logp_side = F.log_softmax(outputs_side_each / args.T, dim=1)
                                loss_distill_side += -torch.mean(torch.sum(soft_target_side * logp_side, dim=1))
                            loss_distill_side = loss_distill_side / args.side_classifier
                        alpha = float(iteration) / float(iteration + 1)
                        loss = (1-alpha) * loss_cls + alpha * (loss_distill_main + loss_distill_side)

                    tg_optimizer.zero_grad()
                    loss.backward()
                    tg_optimizer.step()

                if iteration==start_iter:
                    print('Epoch: %d, LR: %.4f, loss_cls: %.4f' % (epoch, tg_lr_scheduler.get_last_lr()[0], loss_cls.item()))
                else:
                    print('Epoch: %d, LR: %.4f, loss_cls: %.4f, loss_distill_main: %.4f, loss_distill_side: %.4f' % (
                    epoch, tg_lr_scheduler.get_last_lr()[0], loss_cls.item(), loss_distill_main.item(), loss_distill_main.item()))

                # evaluate the val set
                if (epoch + 1) % val_epoch == 0:
                    tg_model.eval()
                    # if iteration>start_iter:
                    #     ## joint classifiers
                    #     #num_old_classes = ref_model.fc.out_features
                    #     tg_model.fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
                    #     tg_model.fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
                    print("##############################################################")
                    # Calculate validation error of model on the original classes:
                    map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
                    # print('Computing accuracy on the original batch of classes...')
                    evalset.test_data = X_valid_ori.astype('uint8')
                    evalset.test_labels = map_Y_valid_ori
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
                    acc_old = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl*(iteration+1))
                    print('Old classes accuracy: {:.2f} %'.format(acc_old))
                    ##
                    indices_test_subset_cur = np.array([i in order[range(iteration * args.nb_cl, (iteration+1) * args.nb_cl)] for i in Y_valid_total])
                    X_valid_cur = X_valid_total[indices_test_subset_cur]
                    Y_valid_cur = Y_valid_total[indices_test_subset_cur]
                    map_Y_valid_cur = np.array([order_list.index(i) for i in Y_valid_cur])
                    # print('Computing accuracy on the original batch of classes...')
                    evalset.test_data = X_valid_cur.astype('uint8')
                    evalset.test_labels = map_Y_valid_cur
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
                    acc_cur = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl*(iteration+1))
                    if acc_cur > acc_best:
                        acc_best = acc_cur
                    print('New classes accuracy: {:.2f} %'.format(acc_cur))
                    # Calculate validation error of model on the cumul of classes:
                    acc = compute_accuracy_WI(tg_model, testloader, 0, args.nb_cl*(iteration+1))
                    print('Total accuracy: {:.2f} %'.format(acc))
                    print('Best accuracy: {:.2f} %'.format(acc_best))
                    print("##############################################################")
                    tg_model.train()
                    ## record accuracy
                    top1_acc_list[n_run, iteration, int((epoch + 1)/val_epoch)-1] = np.array(acc)
                tg_lr_scheduler.step()  # gmpark
########## end of Stage 1

                # Save the val set
                if (epoch + 1) % save_epoch == 0:
                    if not os.path.isdir(ckp_prefix):                                                           
                        os.mkdir(ckp_prefix)
                    ckp_name = os.path.join(ckp_prefix + 'MCD_ResNet32_Model_run_{}_step_{}.pth'.format(n_run, iteration))
                    file = open('{}'.format(ckp_name),'w')
                    torch.save(tg_model.state_dict(), ckp_name)

##################################################################
        # Final save of the results
        print("Save accuracy results for iteration {}".format(iteration))
        ckp_name = os.path.join(ckp_prefix + 'LwF_top1_acc_list.mat')
        sio.savemat(ckp_name, {'accuracy': stage1_acc_list})
        file.close()

##################################################################
print("##############################################################")
print('Final accuracies of each group')
print()
print("stage1_acc_list")
print(stage1_acc_list)
print()
# print(top1_acc_list)
print("##############################################################")
end = time.localtime()

print("Start time : %04d.%02d.%02d %02d:%02d:%02d" % (start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, start.tm_min, start.tm_sec))
print("End time   : %04d.%02d.%02d %02d:%02d:%02d" % (end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))

