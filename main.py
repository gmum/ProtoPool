import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import PrototypeChooser
from utils import mixup_data, find_high_activation_crop
import os
import matplotlib.pyplot as plt
import cv2

from utils import mixup_data, compute_proto_layer_rf_info_v2, compute_rf_prototype



def save_model(model, path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }, path)


def load_model(model, path, device):
    if device.type == 'cuda':
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'\033[0;32mLoad model form: {path}\033[0m')
    return model, checkpoint['epoch']


def adjust_learning_rate(optimizer, rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate


def learn_model(opt: Optional[List[str]]) -> None:
    parser = argparse.ArgumentParser(description='PrototypeGraph')
    parser.add_argument('--data_type', default='birds', choices=['birds', 'cars'])
    parser.add_argument('--data_train', help='Path to train data')
    parser.add_argument('--data_push', help='Path to push data')
    parser.add_argument('--data_test', help='Path to tets data')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='input batch size for training (default: 80)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--push_start', type=int, default=20)
    parser.add_argument('--when_push', type=int, default=2)
    parser.add_argument('--no_cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--num_descriptive', type=int, default=10)
    parser.add_argument('--num_prototypes', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=200)

    parser.add_argument('--arch', type=str, default='resnet34')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--add_on_layers_type', type=str, default='log')
    parser.add_argument('--prototype_activation_function',
                        type=str, default='log')

    parser.add_argument('--use_thresh', action='store_true')
    parser.add_argument('--earlyStopping', type=int, default=None,
                        help='Number of epochs to early stopping')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--results', default='./results',
                        help='Path to dictionary where will be save results.')
    parser.add_argument('--ppnet_path', default=None)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup_time', default=100, type=int)
    parser.add_argument('--gumbel_time', default=10, type=int)
    parser.add_argument('--proto_depth', default=128, type=int)
    parser.add_argument('--last_layer', action='store_true')
    parser.add_argument('--inat', action='store_true')
    parser.add_argument('--mixup_data', action='store_true')
    parser.add_argument('--push_only', action='store_true')
    parser.add_argument('--gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    parser.add_argument('--proto_img_dir', type=str, default='img')
    parser.add_argument('--pp_ortho', action='store_true')
    parser.add_argument('--pp_gumbel', action='store_true')

    if opt is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\033[0;1;31m{device=}\033[0m')

    start_val = 1.3
    end_val = 10 ** 3
    epoch_interval = args.gumbel_time
    alpha = (end_val / start_val) ** 2 / epoch_interval

    def lambda1(epoch): return start_val * np.sqrt(alpha *
                                                   (epoch)) if epoch < epoch_interval else end_val

    clst_weight = 0.8
    sep_weight = -0.08
    tau = 1

    if args.seed is None:  # 1234
        args.seed = np.random.randint(10, 10000, size=1)[0]
    torch.manual_seed(args.seed)
    kwargs = {}
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        kwargs.update({'num_workers': 9, 'pin_memory': True})

    transforms_train_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transforms_push = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.data_type == 'birds':
        train_dataset = datasets.ImageFolder(
            args.data_train,
#            '/shared/sets/datasets/birds/train_birds_augmented/train_birds_augmented/train_birds_augmented/',
            transforms_train_test,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            **kwargs)

        train_push_dataset = datasets.ImageFolder(
            args.data_push,
#            '/shared/sets/datasets/birds/train_birds/train_birds/train_birds/',
            transforms_push,
        )
        train_push_loader = torch.utils.data.DataLoader(
            train_push_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            **kwargs)

        test_dataset = datasets.ImageFolder(
            args.data_test,
#            '/shared/sets/datasets/birds/test_birds/test_birds/test_birds/',
            transforms_train_test,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            **kwargs)

    elif args.data_type == 'cars':
        train_dataset = datasets.ImageFolder(
            args.data_train,
#            '/shared/sets/datasets/stanford_cars/train_cars_augmented/',
            transforms_train_test,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            **kwargs)

        train_push_dataset = datasets.ImageFolder(
            args.data_push,
#            '/shared/sets/datasets/stanford_cars/train_cars/',
            transforms_push,
        )
        train_push_loader = torch.utils.data.DataLoader(
            train_push_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            **kwargs)

        test_dataset = datasets.ImageFolder(
            args.data_test,
#            '/shared/sets/datasets/stanford_cars/test_cars/',
            transforms_train_test,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
            **kwargs)
    else:
        raise ValueError

    model = PrototypeChooser(
        num_prototypes=args.num_prototypes,
        num_descriptive=args.num_descriptive,
        num_classes=args.num_classes,
        use_thresh=args.use_thresh,
        arch=args.arch,
        pretrained=args.pretrained,
        add_on_layers_type=args.add_on_layers_type,
        prototype_activation_function=args.prototype_activation_function,
        proto_depth=args.proto_depth,
        use_last_layer=args.last_layer,
        inat=args.inat,
    )
    if args.ppnet_path:
        model.load_state_dict(torch.load(args.ppnet_path, map_location='cpu')[
                              'model_state_dict'], strict=True)
        print('Successfully loaded ' + args.ppnet_path)

    model.to(device)
    if args.warmup:
        model.features.requires_grad_(False)
        model.last_layer.requires_grad_(True)
        if args.ppnet_path:
            model.add_on_layers.requires_grad_(False)
            model.prototype_vectors.requires_grad_(False)
    if args.checkpoint:
        model, start_epoch = load_model(model, args.checkpoint, device)
    else:
        start_epoch = 0

    warm_optimizer = torch.optim.Adam(
        [{'params': model.add_on_layers.parameters(), 'lr': 3 * args.lr, 'weight_decay': 1e-3},
         {'params': model.proto_presence, 'lr': 3 * args.lr},
         {'params': model.prototype_vectors, 'lr': 3 * args.lr}])
    joint_optimizer = torch.optim.Adam(
        [{'params': model.features.parameters(), 'lr': args.lr / 10, 'weight_decay': 1e-3},
         {'params': model.add_on_layers.parameters(), 'lr': 3 * args.lr,
          'weight_decay': 1e-3},
         {'params': model.proto_presence, 'lr': 3 * args.lr},
         {'params': model.prototype_vectors, 'lr': 3 * args.lr}]
    )
    push_optimizer = torch.optim.Adam(
        [{'params': model.last_layer.parameters(), 'lr': args.lr / 10,
          'weight_decay': 1e-3}, ]
    )
    optimizer = warm_optimizer
    criterion = torch.nn.CrossEntropyLoss()

    info = f'{args.data_type}_descriptive-{args.num_descriptive}_prototypes-{args.num_prototypes}' \
           f'_lr-{args.lr}' \
           f'_{args.arch}_{"True" if args.pretrained else f"No"}' \
           f'_{args.add_on_layers_type}_{args.prototype_activation_function}' \
           f'{"_warmup" if args.warmup else ""}' \
           f'{"_ll" if args.last_layer else ""}' \
           f'{"_mixup" if args.mixup_data else ""}' \
           f'{"_iNaturalist" if args.inat else ""}' \
           f'_seed-{args.seed}' \
           f'_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}'
    path_tensorboard = f'{args.results}/tensorboard/{info}'
    Path(path_tensorboard).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path_tensorboard)
    dir_checkpoint = f'{args.results}/checkpoint/{info}'
    if args.proto_img_dir:
        proto_img_dir = f'{args.results}/img_proto/{info}'
        Path(proto_img_dir).mkdir(parents=True, exist_ok=True)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    

    ####################################
    #          learning model          #
    ####################################
    min_val_loss = np.Inf
    max_val_tst = 0
    epochs_no_improve = 0

    epoch_tqdm = range(start_epoch, args.epochs)
    steps = False

    model_multi = torch.nn.DataParallel(model)

    if not args.push_only:
        print('Model learning')
        for epoch in epoch_tqdm:
            gumbel_scalar = lambda1(epoch) if args.pp_gumbel else 0

            if args.warmup and args.warmup_time == epoch:
                model.features.requires_grad_(True)
                optimizer = joint_optimizer
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.1)
                steps = True
                print("Warm up ends")

            model.train()
            if (epoch + 1) % 8 == 0 and tau > 0.3:
                tau = 0.8 * tau

            ####################################
            #            train step            #
            ####################################
            trn_loss = 0
            trn_tqdm = enumerate(train_loader, 0)
            if epoch > 0:
                for i, (data, label) in trn_tqdm:
                    label_p = label.numpy().tolist()
                    data = data.to(device)
                    label = label.to(device)

                    if args.mixup_data:
                        data, targets_a, targets_b, lam = mixup_data(
                            data, label, 0.5)

                    # ===================forward=====================
                    prob, min_distances, proto_presence = model_multi(
                        data, gumbel_scale=gumbel_scalar)
                    np.savez_compressed(f'{dir_checkpoint}/pp_{epoch * 80 + i}.pth', proto_presence.detach().cpu().numpy())


                    if args.mixup_data:
                        entropy_loss = lam * \
                            criterion(prob, targets_a) + (1 - lam) * \
                            criterion(prob, targets_b)
                    else:
                        entropy_loss = criterion(prob, label)
                    orthogonal_loss = torch.Tensor([0]).cuda()
                    if args.pp_ortho:
                        for c in range(0, model_multi.module.proto_presence.shape[0], 1000):
                            orthogonal_loss_p = \
                                torch.nn.functional.cosine_similarity(model_multi.module.proto_presence.unsqueeze(2)[c:c+1000],
                                                                      model_multi.module.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            orthogonal_loss += orthogonal_loss_p
                        orthogonal_loss = orthogonal_loss / (args.num_descriptive * args.num_classes) - 1

                    proto_presence = proto_presence[label_p]
                    inverted_proto_presence = 1 - proto_presence
                    clst_loss_val = \
                        dist_loss(model, min_distances, proto_presence,
                                  args.num_descriptive)  
                    sep_loss_val = dist_loss(model, min_distances, inverted_proto_presence,
                                             args.num_prototypes - args.num_descriptive)  

                    prototypes_of_correct_class = proto_presence.sum(
                        dim=-1).detach()
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    avg_separation_cost = \
                        torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                                dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)

                    l1_mask = 1 - \
                        torch.t(model.prototype_class_identity).cuda()
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)

                    loss = entropy_loss + clst_loss_val * clst_weight + \
                        sep_loss_val * sep_weight + 1e-4 * l1 + orthogonal_loss 

                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar('train/loss', loss,
                                      epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/entropy', entropy_loss.item(), epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/clst', clst_loss_val.item(), epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/sep', sep_loss_val.item(), epoch * len(train_loader) + i)
                    writer.add_scalar('train/l1', l1.item(),
                                      epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/avg_sep', avg_separation_cost.item(), epoch * len(train_loader) + i)
                    writer.add_scalar(
                        'train/orthogonal_loss', orthogonal_loss.item(), epoch * len(train_loader) + i)
                    trn_loss += loss.item()
                trn_loss /= len(train_loader)
            if steps:
                lr_scheduler.step()

            ####################################
            #          validation step         #
            ####################################
            model_multi.eval()
            tst_loss = np.zeros((args.num_classes, 1))
            prob_leaves = np.zeros((args.num_classes, 1))
            tst_acc, total = 0, 0
            tst_tqdm = enumerate(test_loader, 0)
            with torch.no_grad():
                for i, (data, label) in tst_tqdm:
                    data = data.to(device)
                    label_p = label.detach().numpy().tolist()
                    label = label.to(device)

                    # ===================forward=====================

                    prob, min_distances, proto_presence = model_multi(data, gumbel_scale=gumbel_scalar)

                    loss = criterion(prob, label)
                    entropy_loss = loss

                    orthogonal_loss = 0
                    orthogonal_loss = torch.Tensor([0]).cuda()                                                                                                                                            
                    if args.pp_ortho: 
                        for c in range(0, model_multi.module.proto_presence.shape[0], 1000):
                            orthogonal_loss_p = \
                                torch.nn.functional.cosine_similarity(model_multi.module.proto_presence.unsqueeze(2)[c:c+1000],
                                                                      model_multi.module.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            orthogonal_loss += orthogonal_loss_p
                        orthogonal_loss = orthogonal_loss / (args.num_descriptive * args.num_classes) - 1
                    inverted_proto_presence = 1 - proto_presence

                    l1_mask = 1 - torch.t(model_multi.module.prototype_class_identity).cuda()
                    l1 = (model_multi.module.last_layer.weight * l1_mask).norm(p=1)

                    proto_presence = proto_presence[label_p]
                    inverted_proto_presence = inverted_proto_presence[label_p]
                    clst_loss_val = dist_loss(model_multi.module, min_distances, proto_presence, args.num_descriptive) * clst_weight
                    sep_loss_val = dist_loss(model_multi.module, min_distances, inverted_proto_presence, args.num_prototypes - args.num_descriptive, sep=True) * sep_weight
                    loss = entropy_loss + clst_loss_val + sep_loss_val + orthogonal_loss + 1e-4 * l1
                    tst_loss += loss.item()

                    _, predicted = torch.max(prob, 1)
                    prob_leaves += prob.mean(dim=0).unsqueeze(
                        1).detach().cpu().numpy()
                    true = label
                    tst_acc += (predicted == true).sum()
                    total += label.size(0)

            tst_loss /= len(test_loader)
            tst_acc = tst_acc.item() / total

            ####################################
            #             logger               #
            ####################################

            tst_loss = tst_loss.mean()
            writer.add_scalar('test/acc', tst_acc, epoch)
            writer.add_scalar('test/loss', tst_loss.mean(), epoch)
            writer.add_scalar('test/entropy', entropy_loss.item(), epoch)
            writer.add_scalar('test/clst', clst_loss_val.item(), epoch)
            writer.add_scalar('test/sep', sep_loss_val.item(), epoch)
            writer.add_scalar('test/orthogonal_loss',
                              orthogonal_loss.item(), epoch)
            writer.add_scalar('test/l1', l1.item(), epoch)
            if trn_loss is None:
                trn_loss = loss.mean().detach()
                trn_loss = trn_loss.cpu().numpy() / len(train_loader)
            print(f'Epoch {epoch}|{args.epochs}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
                  f'| acc: {tst_acc:.5f}, orthogonal: {orthogonal_loss.item():.5f} '
                  f'(minimal test-loss: {min_val_loss:.5f}, early stop: {epochs_no_improve}|{args.earlyStopping}) - ')

            ####################################
            #  scheduler and early stop step   #
            ####################################
            if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
                # save the best model
                if tst_acc > max_val_tst:
                    save_model(model_multi.module, f'{dir_checkpoint}/best_model.pth', epoch)

                epochs_no_improve = 0
                if tst_loss.mean() < min_val_loss:
                    min_val_loss = tst_loss.mean()
                if tst_acc > max_val_tst:
                    max_val_tst = tst_acc
            else:
                epochs_no_improve += 1

            if args.use_scheduler:
                # scheduler.step()
                if epochs_no_improve > 5:
                    adjust_learning_rate(optimizer, 0.95)

            if args.earlyStopping is not None and epochs_no_improve > args.earlyStopping:
                print('\033[1;31mEarly stopping!\033[0m')
                break
    ####################################
    #            push step             #
    ####################################
    print('Model push')
    model_multi.eval()

    ####################################
    #          validation step         #
    ####################################
    tst_loss = np.zeros((args.num_classes, 1))
    tst_acc, total = 0, 0
    tst_tqdm = enumerate(test_loader, 0)
    with torch.no_grad():
        for i, (data, label) in tst_tqdm:
            data = data.to(device)
            label = label.to(device)

            # ===================forward=====================
            prob, min_distances, proto_presence = model_multi(data, gumbel_scale=10e3)

            loss = criterion(prob, label)
            entropy_loss = loss

            l1_mask = 1 - torch.t(model_multi.module.prototype_class_identity).cuda()
            l1 = 1e-4 * (model_multi.module.last_layer.weight * l1_mask).norm(p=1)

            loss = entropy_loss + l1
            tst_loss += loss.item()

            _, predicted = torch.max(prob, 1)
            true = label
            tst_acc += (predicted == true).sum()
            total += label.size(0)

        tst_loss /= len(test_loader)
        tst_acc = tst_acc.item() / total
    print(
        f'Before tuning, test loss: {tst_loss.mean():.5f} | acc: {tst_acc:.5f}')

    global_min_proto_dist = np.full(model_multi.module.num_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [model_multi.module.num_prototypes,
         model_multi.module.prototype_shape[1],
         model_multi.module.prototype_shape[2],
         model_multi.module.prototype_shape[3]])

    proto_rf_boxes = np.full(shape=[model.num_prototypes, 6],
                                fill_value=-1)
    proto_bound_boxes = np.full(shape=[model.num_prototypes, 6],
                                        fill_value=-1)

    search_batch_size = train_push_loader.batch_size     

    for push_iter, (search_batch_input, search_y) in enumerate(train_push_loader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''

        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input=search_batch_input, 
                                   start_index_of_search_batch=start_index_of_search_batch,
                                   model=model_multi.module,
                                   global_min_proto_dist=global_min_proto_dist,
                                   global_min_fmap_patches=global_min_fmap_patches,
                                   proto_rf_boxes=proto_rf_boxes,
                                   proto_bound_boxes=proto_bound_boxes,
                                   class_specific=True,
                                   search_y=search_y,
                                   prototype_layer_stride=1,
                                   dir_for_saving_prototypes=proto_img_dir,
                                   prototype_img_filename_prefix='prototype-img',
                                   prototype_self_act_filename_prefix='prototype-self-act',
                                   prototype_activation_function_in_numpy=None)

    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(model_multi.module.prototype_shape))
    model_multi.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    # ===================fine tune=====================

    print('Fine-tuning')
    max_val_tst = 0
    min_val_loss = 10e5
    for tune_epoch in range(25):
        trn_loss = 0
        trn_tqdm = enumerate(train_loader, 0)
        model_multi.train()
        for i, (data, label) in trn_tqdm:
            data = data.to(device)
            label = label.to(device)

            # ===================forward=====================
            if args.mixup_data:
                data, targets_a, targets_b, lam = mixup_data(data, label, 0.5)

            # ===================forward=====================
            prob, min_distances, proto_presence = model_multi(data, gumbel_scale=10e3)

            if args.mixup_data:
                entropy_loss = lam * \
                    criterion(prob, targets_a) + (1 - lam) * \
                    criterion(prob, targets_b)
            else:
                entropy_loss = criterion(prob, label)

            l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
            l1 = 1e-4 * (model_multi.module.last_layer.weight * l1_mask).norm(p=1)

            loss = entropy_loss + l1

            # ===================backward====================
            push_optimizer.zero_grad()
            loss.backward()
            push_optimizer.step()
            trn_loss += loss.item()

            writer.add_scalar('train_push/loss', loss,
                              tune_epoch * len(train_loader) + i)
            writer.add_scalar('train_push/l1', l1.item(),
                              tune_epoch * len(train_loader) + i)

        ####################################
        #          validation step         #
        ####################################
        model_multi.eval()
        tst_loss = np.zeros((args.num_classes, 1))
        tst_acc, total = 0, 0
        tst_tqdm = enumerate(test_loader, 0)
        with torch.no_grad():
            for i, (data, label) in tst_tqdm:
                data = data.to(device)
                label = label.to(device)

                # ===================forward=====================
                prob, min_distances, proto_presence = model_multi(data, gumbel_scale=10e3)

                loss = criterion(prob, label)
                entropy_loss = loss

                l1_mask = 1 - torch.t(model_multi.module.prototype_class_identity).cuda()
                l1 = 1e-4 * (model_multi.module.last_layer.weight * l1_mask).norm(p=1)

                loss = entropy_loss + l1
                tst_loss += loss.item()

                _, predicted = torch.max(prob, 1)
                true = label
                tst_acc += (predicted == true).sum()
                total += label.size(0)

            tst_loss /= len(test_loader)
            tst_acc = tst_acc.item() / total
        ####################################
        #             logger               #
        ####################################

        tst_loss = tst_loss.mean()
        writer.add_scalar('test_push/acc', tst_acc, tune_epoch)
        writer.add_scalar('test_push/loss', tst_loss.mean(), tune_epoch)
        writer.add_scalar('test_push/entropy', entropy_loss.item(), tune_epoch)
        writer.add_scalar('test_push/l1', l1.item(), tune_epoch)
        if trn_loss is None:
            trn_loss = loss.mean().detach()
            trn_loss = trn_loss.cpu().numpy() / len(train_loader)
        print(f'Epoch {tune_epoch}|{5}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
              f'| acc: {tst_acc:.5f}, (minimal test-loss: {min_val_loss:.5f}- ')

        ####################################
        #  scheduler and early stop step   #
        ####################################
        if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
            # save the best model
            if tst_acc > max_val_tst:
                save_model(model_multi.module, f'{dir_checkpoint}/best_model_push.pth', tune_epoch)
            if tst_loss.mean() < min_val_loss:
                min_val_loss = tst_loss.mean()
            if tst_acc > max_val_tst:
                max_val_tst = tst_acc

        if (tune_epoch + 1) % 5 == 0:
            adjust_learning_rate(push_optimizer, 0.95)

    writer.close()
    print('Finished training model. Have nice day :)')


def dist_loss(model, min_distances, proto_presence, top_k, sep=False):
    #         model, [b, p],        [b, p, n],      [scalar]
    max_dist = (model.prototype_shape[1]
                * model.prototype_shape[2]
                * model.prototype_shape[3])
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]
    _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(1, src=torch.ones_like(
        basic_proto), index=idx)  # [b, p]
    inverted_distances, _ = torch.max(
        (max_dist - min_distances) * binarized_top_k, dim=1)  # [b]
    cost = torch.mean(max_dist - inverted_distances)
    return cost


def update_prototypes_on_batch(search_batch_input, start_index_of_search_batch,
                               model,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               class_specific=True,
                               search_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None
                               ):
    model.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        proto_dist_torch = model.prototype_distances(search_batch)
        protoL_input_torch = model.conv_features(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    prototype_shape = model.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    if class_specific:
        map_class_to_prototypes = model.get_map_class_to_prototypes()
        protype_to_img_index_dict = {key: [] for key in range(n_prototypes)}
        # img_y is the image's integer label

        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            [protype_to_img_index_dict[prototype].append(
                img_index) for prototype in map_class_to_prototypes[img_label]]

    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype

            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(protype_to_img_index_dict[j]) == 0:
                continue
            proto_dist_j = proto_dist_[protype_to_img_index_dict[j]][:, j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        batch_min_proto_dist_j = np.amin(proto_dist_j)

        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''

                batch_argmin_proto_dist_j[0] = protype_to_img_index_dict[j][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * \
                prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * \
                prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

           # get the receptive field boundary of the image patch
            # that generates the representation
            # protoL_rf_info = model.proto_layer_rf_info
            layer_filter_sizes, layer_strides, layer_paddings = model.features.conv_info()
            protoL_rf_info = compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings,
                                           prototype_kernel_size=1)
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            original_img_j = (original_img_j - np.min(original_img_j)) / np.max(original_img_j - np.min(original_img_j))
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if model.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + model.epsilon))
            elif model.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PrototypeGraph')
    parser.add_argument('--evaluate', '-e', action='store_true',
                        help='The run evaluation training model')
    args, unknown = parser.parse_known_args()

    learn_model(unknown)

