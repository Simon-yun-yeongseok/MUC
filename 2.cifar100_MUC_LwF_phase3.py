#!/usr/bin/env python
# coding=utf-8
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
import resnet_model
import utils_pytorch
import pandas
from compute_accuracy import compute_accuracy_WI
from compute_accuracy import TEST


global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("cuda is available")
else:
    print("cuda is fail")
    exit()

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--dataset_dir', default='./data/cifar-100-python', type=str)
parser.add_argument('--OOD_dir', default='./data/SVHN', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--nb_cl_fg', default=20, type=int, help='the number of classes in first group')
parser.add_argument('--nb_cl', default=20, type=int, help='Classes per group')
parser.add_argument('--nb_pnum_classes, nb_cl,rotos', default=0, type=int, help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default='MUC_LwF_cifar100', type=str, help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int, help='Epochs')
parser.add_argument('--val_epoch', default=10, type=int, help='Epochs')
parser.add_argument('--T', default=2, type=float, help='Temperature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, help='Beta for distialltion')
parser.add_argument('--resume', default='True', action='store_true', help='resume from checkpoint')
parser.add_argument('--random_seed', default=1988, type=int, help='random seed')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--side_classifier', default=3, type=int, help='multiple classifiers') ##default 3
parser.add_argument('--Stage3_flag', default='True', action='store_true', help='multiple classifiers')
parser.add_argument('--memory_budget', default=2000, type=int, help='Exemplars of old classes')
args = parser.parse_args()

ckp_prefix = './checkpoint/{}/MUC_LwF/step_{}_K_{}/'.format(args.dataset, args.nb_cl, args.side_classifier)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    t = t.to(device)
    return Variable(t, **kwargs)


########################################
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)
train_batch_size       = 128            # Batch size for train(Initial = 128)
test_batch_size        = 100            # Batch size for test
eval_batch_size        = 100            # Batch size for eval
base_lr                = 0.1            # Initial learning rate
lr_strat               = [120, 160, 180]      # Epochs where learning rate gets decreased
lr_factor              = 0.1            # Learning rate decrease factor
custom_weight_decay    = 5e-4           # Weight Decay
custom_momentum        = 0.9            # Momentum
epochs                 = 20            # initial = 200
val_epoch              = 5             # evaluate the model in every val_epoch(initial = 10)
save_epoch             = 10             # save the model in every save_epoch(initial = 50)
np.random.seed(args.random_seed)        # Fix the random seed
print(args)
Stage1_flag = True  # Train new model and new classifier
Stage3_flag = True  # Train side classifiers with Maximum Classifier Discrepancy  Initial : True
########################################

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
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
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=True, transform=transform_test)
evalset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                       download=False, transform=transform_test)

# save accuracy
top1_acc_list = np.zeros((args.nb_runs, int(args.num_classes/args.nb_cl), int(epochs/val_epoch)))
old_val_list = np.zeros(int(args.num_classes/args.nb_cl))
old_val_list_sub = np.zeros(int(args.num_classes/args.nb_cl))
stage1_acc_list = np.zeros((int(args.num_classes/args.nb_cl), int(args.num_classes/args.nb_cl)))

stage3_old_val_list = np.zeros(int(args.num_classes/args.nb_cl))
stage3_old_val_list_sub = np.zeros(int(args.num_classes/args.nb_cl))
stage3_acc_list = np.zeros((int(args.num_classes/args.nb_cl), int(args.num_classes/args.nb_cl)))
top1_acc_cur_list = np.zeros((args.nb_runs, int(args.num_classes/args.nb_cl), int(epochs/val_epoch)))

X_train_total = np.array(trainset.data)
Y_train_total = np.array(trainset.targets)
X_valid_total = np.array(testset.data)     
Y_valid_total = np.array(testset.targets)

## Load unlabeled data from SVHN
svhn_data = torchvision.datasets.SVHN(root=args.OOD_dir, download=True, transform=transform_train)
svhn_num = svhn_data.data.shape[0]
svhn_data_copy = svhn_data.data
svhn_labels_copy = svhn_data.labels

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
        # Prepare the training data for the current batch of classes(total class(100)/group class(20))
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

        X_train_sub = torch.tensor(X_train, dtype=torch.float32)
        map_Y_train_sub = torch.tensor(map_Y_train, dtype=torch.long)
        X_valid_sub = torch.tensor(X_valid, dtype=torch.float32)
        map_Y_valid_sub = torch.tensor(map_Y_valid, dtype=torch.long)

        
        train_subset = torch.utils.data.TensorDataset(X_train_sub, map_Y_train_sub)
        test_subset = torch.utils.data.TensorDataset(X_valid_sub, map_Y_valid_sub)
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=train_batch_size, shuffle=False, num_workers=2)
        testloader = torch.utils.data.DataLoader(test_subset, batch_size=test_batch_size, shuffle=False, num_workers=2)

        print('Min and Max of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Min and Max of valid labels: {}, {}'.format(min(map_Y_valid), max(map_Y_valid)))
        
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
            tg_model = resnet_model.resnet32_custom(num_classes=args.nb_cl, side_classifier=args.side_classifier)
            tg_model = tg_model.to(device)
            ref_model = None
            num_old_classes = 0
            for param in tg_model.parameters():
                param.requires_grad = True
        else:
            #increment classes
            ref_model = copy.deepcopy(tg_model)
            ref_model = ref_model.to(device)
            for param in ref_model.parameters():
                param.requires_grad = False

            ## new main classifier
            num_old_classes = ref_model.fc.out_features
            in_features = ref_model.fc.in_features
            new_fc = nn.Linear(in_features, args.nb_cl*(iteration+1)).cuda()
            new_fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
            new_fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
            tg_model.fc = new_fc
            for param in tg_model.fc.parameters():
                param.requires_grad = True
            
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
            # tg_optimizer = optim.SGD(update_params, lr=base_lr, weight_decay=custom_weight_decay)
            tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
            cls_criterion = nn.CrossEntropyLoss()
            cls_criterion.to(device)
            for epoch in range(epochs):
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    if args.cuda:
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    
                    if iteration == start_iter:
                        outputs = tg_model(inputs, side_fc=False)
                        loss_cls = cls_criterion(outputs[:, num_old_classes:(num_old_classes + args.nb_cl)], targets)
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
                tg_lr_scheduler.step()

                if iteration==start_iter:
                    print('Epoch: %d, LR: %.4f, loss_cls: %.6f' % (epoch, tg_lr_scheduler.get_last_lr()[0], loss_cls.item()))
                    #print(acts)
                else:
                    print('Epoch: %d, LR: %.4f, loss_cls: %.6f, loss_distill_main: %.6f' % (epoch, 
                    tg_lr_scheduler.get_last_lr()[0], loss_cls.item(), loss_distill_main.item()))
                
                # evaluate the val set
                if (epoch + 1) % val_epoch == 0:
                    tg_model.eval()
                    print("##############################################################")
                    # Calculate validation accuracy of model on the current classes:
                    for i in range(iteration):
                        print("iteration :{}".format(i))
                        indices_valid_subset_old = np.array([j in order[range(i * args.nb_cl, (i+1) * args.nb_cl)] for j in Y_valid_total])
                        X_valid_old = X_valid_total[indices_valid_subset_old]
                        Y_valid_old = Y_valid_total[indices_valid_subset_old]
                        map_Y_valid_old = np.array([order_list.index(j) for j in Y_valid_old]) 
                        X_eval_sub = torch.tensor(X_valid_old, dtype=torch.float32)
                        map_Y_eval_sub = torch.tensor(map_Y_valid_old, dtype=torch.long)
                        print('Min and Max of mapped eval old labels: {}, {}'.format(min(map_Y_valid_old), max(map_Y_valid_old)))

                        eval_subset = torch.utils.data.TensorDataset(X_eval_sub, map_Y_eval_sub)
                        evalloader = torch.utils.data.DataLoader(eval_subset, batch_size=train_batch_size, shuffle=False, num_workers=2)
                        acc_old = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl) 
                        print('Old classes(group{}) accuracy: {:.2f} %'.format((iteration),(acc_old)))
                        old_val_list_sub[i] = np.array(acc_old)
                        
                    if epoch+1 == val_epoch:
                        old_val_list = copy.deepcopy(old_val_list_sub)
                    else:
                        old_val_list = np.maximum(old_val_list,old_val_list_sub)
                    
                    for i in range(iteration +1):
                        stage1_acc_list[iteration, i] = old_val_list[i]

                    # Calculate validation accuracy of model on the current classes:
                    # print('Computing accuracy on the original batch of classes...')
                    indices_test_subset_cur = np.array([i in order[range(iteration * args.nb_cl, (iteration+1) * args.nb_cl)] for i in Y_valid_total])
                    X_valid_cur = X_valid_total[indices_test_subset_cur]
                    Y_valid_cur = Y_valid_total[indices_test_subset_cur]
                    map_Y_valid_cur = np.array([order_list.index(i) for i in Y_valid_cur]) 
                    X_eval_sub = torch.tensor(X_valid_cur, dtype=torch.float32)
                    map_Y_eval_sub = torch.tensor(map_Y_valid_cur, dtype=torch.long)

                    eval_subset = torch.utils.data.TensorDataset(X_eval_sub, map_Y_eval_sub)
                    evalloader = torch.utils.data.DataLoader(eval_subset, batch_size=train_batch_size, shuffle=False, num_workers=2)
                    acc_cur_sub = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl)
                    
                    if epoch+1 == val_epoch:
                        acc_cur = copy.deepcopy(acc_cur_sub)
                    else:
                        acc_cur = np.maximum(acc_cur,acc_cur_sub)
                    print('Current classes accuracy: {:.2f} %'.format(acc_cur))
                    
                    top1_acc_cur_list[n_run, iteration, int((epoch + 1)/val_epoch)-1] = np.array(acc_cur) ####

                    # print(top1_acc_cur_list)
                    # print("##############################################################")
                    
                    # Calculate total accuracy of current model:
                    acc = compute_accuracy_WI(tg_model, testloader, 0, args.nb_cl*(iteration+1))
                    top1_acc_list[n_run, iteration, int((epoch + 1)/val_epoch)-1] = np.array(acc)
                    # print('Total accuracy: {:.2f} %'.format(acc))
                    # stage1_acc_list[iteration, iteration] = np.array(acc)
                    stage1_acc_list[iteration, iteration] = np.array(acc_cur)

                    tg_model.train()
                    print(stage1_acc_list)
                    # print(top1_acc_cur_list)
                    # print()
                    # print(top1_acc_list)
                    print("##############################################################")
                
                # Save the val set
                if (epoch + 1) % save_epoch == 0:
                    if not os.path.isdir(ckp_prefix):                                                           
                        os.mkdir(ckp_prefix)
                    ckp_name = os.path.join(ckp_prefix + 'MCD_ResNet32_Model_run_{}_step_{}.pth'.format(n_run, iteration))
                    file = open('{}'.format(ckp_name),'w')
                    torch.save(tg_model.state_dict(), ckp_name)

                    
########## Stage 3: Maximum Classifier Discrepancy for each iteration #################
        if Stage3_flag is True:
            print("Stage 3: Train Side Classifiers with Maximum Classifier Discrepancy for iteration {}".format(iteration))
            ##
            stage3_model = copy.deepcopy(tg_model)
            start_index = args.nb_cl * args.side_classifier * iteration
            # print("Initialize Side Classifiers with Main Classifier")
            # for i in range(args.side_classifier):
            #     stage3_model.fc_side.weight.data[(start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))] = stage3_model.fc.weight.data[num_old_classes:]
            #     stage3_model.fc_side.bias.data[(start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))] = stage3_model.fc.bias.data[num_old_classes:]
            stage3_model = stage3_model.to(device)
            stage3_model.eval()
            stage3_model.fc_side.train()
            ## fix feature extractor and main classifier
            for n, p in stage3_model.named_parameters():
                if 'fc_side' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            stage3_lr_strat = [40, 60, 70]
            stage3_epochs = 80
            stage3_params = list(stage3_model.fc_side.parameters())
            stage3_optimizer = optim.SGD(stage3_params, lr=base_lr, momentum=custom_momentum,weight_decay=custom_weight_decay)
            stage3_lr_scheduler = lr_scheduler.MultiStepLR(stage3_optimizer, milestones=stage3_lr_strat, gamma=lr_factor)
            cls_criterion = nn.CrossEntropyLoss()
            cls_criterion.to(device)
            ## Train
            for stage3_epoch in range(stage3_epochs):
                # select a subset of SVHN data
                idx = torch.randperm(svhn_num)
                svhn_data_copy = svhn_data_copy[idx]
                svhn_labels_copy = svhn_labels_copy[idx]
                svhn_data.data = svhn_data_copy[0:len(map_Y_train_sub)]
                svhn_data.labels = svhn_labels_copy[0:len(map_Y_train_sub)]
                svhn_loader = torch.utils.data.DataLoader(dataset=svhn_data, batch_size=train_batch_size, shuffle=True, num_workers=2)
                for ((batch_idx, (inputs, targets)), (batch_idx_unlabel, (inputs_unlabel, targets_unlabel))) in zip(
                        enumerate(trainloader), enumerate(svhn_loader)):
                    if args.cuda:
                        inputs, targets, inputs_unlabel, targets_unlabel = inputs.to(device), targets.to(
                            device), inputs_unlabel.to(device), targets_unlabel.to(device)

                    targets = targets - args.nb_cl * iteration
                    loss_cls = 0
                    outputs = stage3_model(inputs, side_fc=True)
                    for i in range(args.side_classifier):
                        loss_cls += cls_criterion(outputs[:, (start_index + args.nb_cl * i):(start_index + args.nb_cl * (i + 1))], targets)

                    ## discrepancy loss
                    outputs_unlabel = stage3_model(inputs_unlabel, side_fc=True)
                    loss_discrepancy = 0
                    for iter_1 in range(args.side_classifier):
                        outputs_unlabel_1 = outputs_unlabel[:, (start_index + args.nb_cl * iter_1):(start_index + args.nb_cl * (iter_1 + 1))]
                        outputs_unlabel_1 = F.softmax(outputs_unlabel_1, dim=1)
                        for iter_2 in range(iter_1 + 1, args.side_classifier):
                            outputs_unlabel_2 = outputs_unlabel[:, (start_index + args.nb_cl * iter_2):(start_index + args.nb_cl * (iter_2 + 1))]
                            outputs_unlabel_2 = F.softmax(outputs_unlabel_2, dim=1)
                            #loss_discrepancy += torch.mean(F.relu(1.0 - torch.sum(torch.abs(outputs_unlabel_1 - outputs_unlabel_2), 1)))
                            loss_discrepancy += torch.mean(torch.mean(torch.abs(outputs_unlabel_1 - outputs_unlabel_2), 1))
                    loss = loss_cls - loss_discrepancy

                stage3_optimizer.zero_grad()
                loss.backward()
                stage3_optimizer.step()
                stage3_lr_scheduler.step()

                print('Epoch: %d, LR: %.4f, loss_cls: %.4f, loss_discrepancy: %.4f' % (
                    stage3_epoch, stage3_lr_scheduler.get_lr()[0], loss_cls.item() / args.side_classifier, loss_discrepancy.item()))

                # evaluate the val set
                if (stage3_epoch + 1) % 10 == 0:
                    stage3_model.fc_side.eval()
                    print("##############################################################")
                    for i in range(iteration):
                        print("iteration :{}".format(i))
                        indices_valid_subset_current = np.array([j in order[range(i * args.nb_cl, (i+1) * args.nb_cl)] for j in Y_valid_total])
                        X_valid_current = X_valid_total[indices_valid_subset_current]
                        Y_valid_current = Y_valid_total[indices_valid_subset_current]
                        map_Y_valid_current = np.array([order_list.index(j) for j in Y_valid_current]) 
                        X_eval_side_sub = torch.tensor(X_valid_current, dtype=torch.float32)
                        map_Y_eval_side_sub = torch.tensor(map_Y_valid_current, dtype=torch.long)
                        print('Min and Max of mapped eval current labels: {}, {}'.format(min(map_Y_valid_old), max(map_Y_valid_old)))
                        eval_side_subset = torch.utils.data.TensorDataset(X_eval_side_sub, map_Y_eval_side_sub)
                        evalloader = torch.utils.data.DataLoader(eval_side_subset, batch_size=train_batch_size, shuffle=False, num_workers=2)
                        # acc_old = compute_accuracy_WI(stage3_model, evalloader, 0, args.nb_cl)
                        stage3_acc_old = compute_accuracy_Version1(stage3_model, evalloader, args.nb_cl, args.side_classifier, iteration) 
                        print('Old classes(group{}) accuracy: {:.2f} %'.format((iteration),(stage_acc_old)))
                        # print('Maximum Classifier Discrepancy accuracy: {:.2f} %'.format(acc))
                        print("##############################################################")
                        stage3_old_val_list_sub[i] = np.array(stage3_acc_old)
                        
                    if epoch+1 == val_epoch:
                        stage3_old_val_list = copy.deepcopy(stage3_old_val_list_sub)
                    else:
                        stage3_old_val_list = np.maximum(stage3_old_val_list,stage3_old_val_list_sub)
                    
                    for i in range(iteration +1):
                        stage3_acc_list[iteration, i] = stage3_old_val_list[i]
                    stage3_model.fc_side.train()

                if (stage3_epoch + 1) % 40 == 0:
                    if not os.path.isdir(ckp_prefix):                                                           
                        os.mkdir(ckp_prefix)
                    ckp_name = os.path.join(ckp_prefix + 'MCD_ResNet32_Model_run_{}_step_{}.pth').format(n_run, iteration)
                    torch.save(stage3_model.state_dict(), ckp_name)

            tg_model.fc_side.weight.data[start_index:] = stage3_model.fc_side.weight.data[start_index:]
            tg_model.fc_side.bias.data[start_index:] = stage3_model.fc_side.bias.data[start_index:]
########## end of Stage 3

##################################################################
        # Final save of the results
        print("Save accuracy results for iteration {}".format(iteration))
        ckp_name = os.path.join(ckp_prefix + 'LwF_top1_acc_list_K={}.mat'.format(args.side_classifier))
        sio.savemat(ckp_name, {'accuracy': stage1_acc_list})
        file.close()
##################################################################



print("done!!")
