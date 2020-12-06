import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
from torchvision import transforms, datasets, version

print('Using pytorch version: {}'.format(torch.__version__))
print('Using torchvision version: {}'.format(version.__version__))

import torch.nn as nn
from torch.utils.data.sampler import *
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.inception import inception_v3
from torchvision.utils import make_grid

"""
hack to deal with vagaries of Colab and Google Drive in batch mode
"""

"""
import shutil

def rreplace(s, old, new='', occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)


script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = script_dir.replace(os.sep, '/') + '/'

postfixed_files = [_file for _file in os.listdir(script_dir) if
                   os.path.splitext(_file)[0].endswith(' (1)')]
if postfixed_files:
    print('postfixed_files: {}'.format(postfixed_files))

    for _file in postfixed_files:
        _dst_file = rreplace(_file, ' (1)', '')
        _src_path = os.path.join(script_dir, _file)
        _dst_path = os.path.join(script_dir, _dst_file)

        print('{} --> {}'.format(_file, _dst_file))
        shutil.move(_src_path, _dst_path)
"""

from A5_submission import SharedNet, Discriminator, Classifier, Generator, CompositeLoss
from A5_submission import TrainParams, SaveCriteria
from A5_utils import REAL_LABEL, FAKE_LABEL, REAL_UNLABELED, FAKE_UNLABELED, print_stats
from A5_utils import get_num_params, PartiallyLabeled, compute_fid, compute_inception_score, GANLosses, InceptionV3


class A5_Params:
    """


    :ivar gen_latent_size: Size of latent vector used as generator input

    :ivar fid_dims: which feature map to use for FID

    :ivar total_split: what fraction of training data to use; working on small subsets can be useful for debugging

    :ivar train_gen_metrics: toggle computing generator metrics FID and IS during training;
    disabling can speed up validation

    :ivar eval_gen_metrics: toggle computing generator metrics FID and IS during validation;
    disabling can speed up validation
    """

    def __init__(self):
        self.use_cuda = 1

        self.train_split = 0.76
        self.labeled_split = 0.2

        self.gen_latent_size = 100
        self.fid_dims = 2048

        self.train = TrainParams()

        """useful for debugging"""
        self.total_split = 1
        self.train_gen_metrics = 0
        self.eval_gen_metrics = 0

        """
        Module specific parameters
        """
        self.shared = SharedNet.Params()
        self.dscr = Discriminator.Params()
        self.cls = Classifier.Params()
        self.gen = Generator.Params()
        self.loss = CompositeLoss.Params()


def evaluate(all_modules, dataloader, composite_loss, inception_model, fid_model, upsample,
             device, params, n_classes=10):
    """

    :param nn.ModuleList all_modules:
    :param dataloader:
    :param CompositeLoss composite_loss:
    :param int vis:
    :param torch.device  device:
    :param A5_Params  params:
    :return:
    """
    all_modules.eval()

    shared_net, discriminator, \
    classifier, generator = all_modules  # type: SharedNet, Discriminator, Classifier, Generator

    n_batches = 0

    real_total = 0
    real_correct = 0

    fake_total = 0
    fake_correct = 0

    total_fid = 0
    total_inception_score = 0

    total_time = 0

    loss = GANLosses()
    loss_item = GANLosses()
    total_loss = GANLosses()
    mean_loss = GANLosses()

    fake_images = fake_images_grid = None
    real_images = real_images_grid = None

    print('evaluating...')

    n_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, (real_images, real_labels) in tqdm(enumerate(dataloader), total=n_batches):

            real_images = real_images.to(device)

            real_labels = real_labels.to(device)  # 0 to 9
            fake_labels = real_labels + n_classes  # 10 to 19

            """update discriminator  / classifier with real images"""
            dscr_real_labels = torch.full((real_labels.size(0),), REAL_LABEL, dtype=torch.float, device=device)
            dscr_fake_labels = torch.full((real_labels.size(0),), FAKE_LABEL, dtype=torch.float, device=device)

            noise = torch.randn(real_labels.size(0), params.gen_latent_size, 1, 1, device=device)

            start_t = time.time()

            shared_out_real = shared_net(real_images)
            dscr_out_real = discriminator(shared_out_real)
            cls_out_real = classifier(shared_out_real)

            """update discriminator  / classifier with fake images"""
            """Generate fake image batch with generator"""
            fake_images = generator(noise)

            shared_out_fake = shared_net(fake_images)
            dscr_out_fake = discriminator(shared_out_fake)
            cls_out_fake = classifier(shared_out_fake)

            """time only forward passes through nets"""
            end_t = time.time()

            test_time = end_t - start_t
            total_time += test_time

            """compute losses"""
            loss.dscr_real = discriminator.get_loss(dscr_out_real, dscr_real_labels)
            loss.cls_real = classifier.get_loss(cls_out_real, real_labels)
            loss.cmp_real = composite_loss(loss.dscr_real, loss.cls_real)

            loss.dscr_fake = discriminator.get_loss(dscr_out_fake, dscr_fake_labels)
            loss.cls_fake = classifier.get_loss(cls_out_fake, fake_labels)
            loss.cmp_fake = composite_loss(loss.dscr_fake, loss.cls_fake)

            """total loss for discriminator / classifier"""
            loss.cmp = loss.cmp_real + loss.cmp_fake

            """fake labels are real for generator loss"""
            loss.dscr_gen = discriminator.get_loss(dscr_out_fake, dscr_real_labels)
            loss.cls_gen = classifier.get_loss(cls_out_fake, real_labels)
            loss.cmp_gen = composite_loss(loss.dscr_gen, loss.cls_gen)

            """classification accuracy for real images"""
            _, pred_real = torch.max(cls_out_real.data, 1)
            pred_real = pred_real.squeeze()
            _correct = pred_real.eq(real_labels).sum().item()
            _total = real_labels.size(0)
            real_total += _total
            real_correct += _correct

            """classification accuracy for fake images"""
            _, pred_fake = torch.max(cls_out_fake.data, 1)
            pred_fake = pred_fake.squeeze()
            fake_total += fake_labels.size(0)
            fake_correct += pred_fake.eq(fake_labels).sum().item()

            """Generator metrics"""
            if params.eval_gen_metrics:
                """resize images to be compatible with inception"""
                real_images_up = upsample(real_images.detach())
                fake_images_up = upsample(fake_images.detach())

                inception_score = compute_inception_score(fake_images_up, inception_model)
                inception_score = inception_score[0]
                fid = compute_fid([real_images_up, fake_images_up], fid_model, device=device, dims=params.fid_dims)

                total_inception_score += inception_score
                total_fid += fid

            total_loss_dict = total_loss.__dict__
            for loss_type in loss.__dict__:
                loss_item.__dict__[loss_type] = loss.__dict__[loss_type]
                total_loss_dict[loss_type] += loss_item.__dict__[loss_type]

            n_batches += 1

    for loss_type in loss.__dict__:
        mean_loss.__dict__[loss_type] = total_loss.__dict__[loss_type] / n_batches

    if real_images is not None:
        real_images_grid = make_grid(real_images, padding=2, normalize=True)

    if fake_images is not None:
        fake_images_grid = make_grid(fake_images, padding=2, normalize=True)

    """mean classification accuracy for real, fake and real+fake"""
    real_acc = 100. * real_correct / real_total
    fake_acc = 100. * fake_correct / fake_total
    total_acc = 100. * (real_correct + fake_correct) / (real_total + fake_total)

    if params.eval_gen_metrics:
        total_inception_score /= n_batches
        total_fid /= n_batches
    else:
        total_inception_score = total_fid = -1

    return mean_loss, (real_acc, fake_acc, total_acc), \
           (total_inception_score, total_fid, real_images_grid, fake_images_grid), total_time


def main():
    """number of classes in FMNIST dataset"""
    n_classes = 10

    params = A5_Params()

    # optional command line argument parsing
    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    # init device
    if params.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.cuda.FloatTensor
        print('Training on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        dtype = torch.FloatTensor
        print('Training on CPU')

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR10('data', train=True, download=True, transform=transform)

    test_set = datasets.CIFAR10('data', train=False, download=True, transform=transform)

    valid_set = datasets.CIFAR10('data', train=True, download=True, transform=transform)

    train_params = params.train

    num_train = int(len(train_set) * params.total_split)
    indices = list(range(num_train))
    split = int(np.floor(params.train_split * num_train))

    train_idx, valid_idx = indices[:split], indices[split:]
    train_set = PartiallyLabeled(train_set, train_idx, labeled_percent=params.labeled_split)

    print('Training samples: {}\n'
          'Validation samples: {}\n'
          'Labeled training samples: {}'
          ''.format(
        len(train_idx),
        len(valid_idx),
        train_set.n_labeled_data
    ))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_params.batch_size, sampler=train_sampler,
                                                   num_workers=train_params.n_workers)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=24, sampler=valid_sampler,
                                                   num_workers=train_params.n_workers)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False,
                                                  num_workers=train_params.n_workers)

    # create all_modules
    shared_net = SharedNet(params.shared).to(device)  # type: SharedNet
    discriminator = Discriminator(params.dscr).to(device)  # type: Discriminator
    classifier = Classifier(params.cls).to(device)  # type: Classifier
    generator = Generator(params.gen, params.gen_latent_size).to(device)  # type: Generator
    composite_loss = CompositeLoss(device, params.loss).to(device)

    assert isinstance(shared_net, nn.Module), 'SharedNet must be an instance of nn.Module'
    assert isinstance(discriminator, nn.Module), 'Discriminator must be an instance of nn.Module'
    assert isinstance(classifier, nn.Module), 'Classifier must be an instance of nn.Module'
    assert isinstance(generator, nn.Module), 'Generator must be an instance of nn.Module'
    assert isinstance(composite_loss, nn.Module), 'CompositeLoss must be an instance of nn.Module'

    n_shared_params = get_num_params(shared_net)
    n_discriminator_params = get_num_params(discriminator)
    n_classifier_params = get_num_params(classifier)

    assert n_shared_params >= n_discriminator_params and n_shared_params >= n_classifier_params, \
        "Discriminator and classifier must have at least half their parameters shared"

    all_modules = nn.ModuleList((shared_net, discriminator, classifier, generator))  # type: nn.ModuleList

    # init weights
    shared_net.init_weights()
    discriminator.init_weights()
    generator.init_weights()
    classifier.init_weights()

    discriminator_opt = discriminator.get_optimizer(nn.ModuleList((shared_net, discriminator, classifier)))
    generator_opt = generator.get_optimizer(generator)

    weights_dir = os.path.dirname(train_params.weights_path)
    weights_name = os.path.basename(train_params.weights_path)

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    start_epoch = 0

    mean_loss = GANLosses()
    eval_metrics = list(mean_loss.__dict__.keys()) + ['total_acc', 'real_acc', 'fake_acc', 'fid', 'is']
    data_types = ['train', 'valid']

    status_df = pd.DataFrame(
        np.zeros((len(data_types), len(eval_metrics)), dtype=np.float32),
        index=data_types,
        columns=eval_metrics,
    )

    # load weights
    if train_params.load_weights:
        matching_ckpts = [k for k in os.listdir(weights_dir) if
                          os.path.isfile(os.path.join(weights_dir, k)) and
                          k.startswith(weights_name)]
        if not matching_ckpts:
            msg = 'No checkpoints found matching {} in {}'.format(weights_name, weights_dir)
            if train_params.load_weights == 1:
                raise IOError(msg)
            print(msg)
        else:
            matching_ckpts.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

            weights_path = os.path.join(weights_dir, matching_ckpts[-1])

            chkpt = torch.load(weights_path, map_location=device)  # load checkpoint

            print('Loading weights from: {} with:\n'
                  '\tcriterion: {}\n'
                  '\tepoch: {}\n'
                  '\ttimestamp: {}\n'.format(
                weights_path,
                chkpt['criterion'],
                chkpt['epoch'],
                chkpt['timestamp']))

            status_df = chkpt['status_df']

            print('stats:')
            print_stats(status_df)
            print()

            shared_net.load_state_dict(chkpt['shared_net'])
            discriminator.load_state_dict(chkpt['discriminator'])
            generator.load_state_dict(chkpt['generator'])
            classifier.load_state_dict(chkpt['classifier'])
            generator_opt.load_state_dict(chkpt['generator_opt'])
            discriminator_opt.load_state_dict(chkpt['discriminator_opt'])

            start_epoch = chkpt['epoch'] + 1
    else:
        print('Training from scratch')

    if params.train_gen_metrics or params.eval_gen_metrics:
        inception_model = inception_v3(pretrained=True, transform_input=False, init_weights=False).type(dtype)
        """custom inception for FID"""
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[params.fid_dims]
        fid_model = InceptionV3(output_blocks=[block_idx], resize_input=False).to(device)
        """needed to resize images to be compatible with inception"""
        upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).type(dtype)
    else:
        inception_model = fid_model = upsample = None

    if train_params.load_weights != 1:
        """
        continue training
        """

        writer = SummaryWriter(log_dir=params.train.tb_path)
        print(f'Saving tensorboard summary to: {params.train.tb_path}')

        """
        training steps from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        """
        iter_id = 0

        """decide when to save weights"""
        save_criteria = SaveCriteria(status_df)

        if not params.train_gen_metrics:
            print('Generator metrics computation is disabled during training')

        if not params.eval_gen_metrics:
            print('Generator metrics computation is disabled during evaluation')

        for epoch in range(start_epoch, train_params.n_epochs):
            all_modules.train()

            real_total = 0
            real_correct = 0

            fake_total = 0
            fake_correct = 0

            batch_idx = 0

            train_fid = 0
            train_inception_score = 0

            total_loss = GANLosses()
            loss = GANLosses()
            loss_item = GANLosses()

            n_batches = len(train_dataloader)

            for batch_idx, (real_images, real_labels, is_labeled) in tqdm(enumerate(train_dataloader), total=n_batches):
                real_images = real_images.to(device)

                real_labels = real_labels.to(device)  # 0 to 9
                fake_labels = real_labels + n_classes  # 10 to 19

                is_labeled = is_labeled.squeeze()
                n_labeled = np.count_nonzero(is_labeled.detach().numpy())

                """remove labels for unlabeled images"""
                is_not_labeled = np.logical_not(is_labeled.cpu().numpy()).squeeze()
                real_labels[is_not_labeled] = REAL_UNLABELED
                fake_labels[is_not_labeled] = FAKE_UNLABELED

                """""""""""""""""""""""""""""""""""""""""""""""""""""
                train discriminator / classifier
                """""""""""""""""""""""""""""""""""""""""""""""""""""
                shared_net.zero_grad()
                discriminator.zero_grad()
                classifier.zero_grad()

                """update discriminator  / classifier with real images"""
                dscr_real_labels = torch.full((real_labels.size(0),), REAL_LABEL, dtype=torch.float, device=device)

                shared_out_real = shared_net(real_images)
                dscr_out_real = discriminator(shared_out_real)
                cls_out_real = classifier(shared_out_real)

                loss.dscr_real = discriminator.get_loss(dscr_out_real, dscr_real_labels)
                loss.cls_real = classifier.get_loss(cls_out_real, real_labels, is_labeled, n_labeled)
                loss.cmp_real = composite_loss(loss.dscr_real, loss.cls_real)

                """compute gradients for the real batch"""
                loss.cmp_real.backward()

                """update discriminator  / classifier with fake images"""
                dscr_fake_labels = torch.full((real_labels.size(0),), FAKE_LABEL, dtype=torch.float, device=device)

                noise = torch.randn(real_labels.size(0), params.gen_latent_size, 1, 1, device=device)

                """Generate fake image batch with generator"""
                fake_images = generator(noise)

                """we do not want generator gradients to be computed"""
                fake_images_no_grad = fake_images.detach()

                shared_out_fake = shared_net(fake_images_no_grad)
                dscr_out_fake = discriminator(shared_out_fake)
                cls_out_fake = classifier(shared_out_fake)

                loss.dscr_fake = discriminator.get_loss(dscr_out_fake, dscr_fake_labels)
                loss.cls_fake = classifier.get_loss(cls_out_fake, fake_labels, is_labeled, n_labeled)
                loss.cmp_fake = composite_loss(loss.dscr_fake, loss.cls_fake)

                """Add the gradients from the fake batch to the real one"""
                loss.cmp_fake.backward()

                """total loss for discriminator / classifier"""
                loss.cmp = loss.cmp_real + loss.cmp_fake

                """update discriminator and classifier parameters"""
                discriminator_opt.step()

                """""""""""""""""""""""""""""""""""""""""""""""""""""
                train generator
                """""""""""""""""""""""""""""""""""""""""""""""""""""
                generator.zero_grad()

                """Since we just updated the discriminator, perform another forward pass of all-fake batch through it"""
                shared_out_gen = shared_net(fake_images)
                dscr_out_gen = discriminator(shared_out_gen)
                cls_out_gen = classifier(shared_out_gen)

                """fake labels are real for generator loss"""
                loss.dscr_gen = discriminator.get_loss(dscr_out_gen, dscr_real_labels)
                loss.cls_gen = classifier.get_loss(cls_out_gen, real_labels, is_labeled, n_labeled)
                loss.cmp_gen = composite_loss(loss.dscr_gen, loss.cls_gen)

                loss.cmp_gen.backward()

                generator_opt.step()

                """""""""""""""""""""""""""""""""""""""""""""""""""""
                collect train statistics and add to tensorboard log
                """""""""""""""""""""""""""""""""""""""""""""""""""""
                total_loss_dict = total_loss.__dict__
                for loss_type in loss.__dict__:
                    loss_item.__dict__[loss_type] = loss.__dict__[loss_type]
                    total_loss_dict[loss_type] += loss_item.__dict__[loss_type]
                    writer.add_scalar(f'train_iter_loss/{loss_type}', loss_item.__dict__[loss_type], iter_id)

                """classification accuracy for real images"""
                _, pred_real = torch.max(cls_out_real.data, 1)
                pred_real = pred_real.squeeze()
                real_total += real_labels.size(0)
                real_correct += pred_real.eq(real_labels).sum().item()

                """classification accuracy for fake images"""
                _, pred_fake = torch.max(cls_out_fake.data, 1)
                pred_fake = pred_fake.squeeze()
                fake_total += fake_labels.size(0)
                fake_correct += pred_fake.eq(fake_labels).sum().item()

                if params.train_gen_metrics:
                    """resize images to be compatible with inception"""
                    real_images_up = upsample(real_images.detach())
                    fake_images_up = upsample(fake_images.detach())

                    inception_score = compute_inception_score(fake_images_up, inception_model)
                    """need only the mean KL Divergence"""
                    inception_score = inception_score[0]
                    fid = compute_fid([real_images_up, fake_images_up], fid_model, device=device, dims=params.fid_dims)

                    train_inception_score += inception_score
                    train_fid += fid

                    writer.add_scalar('train_iter/inception_score', inception_score, iter_id)
                    writer.add_scalar('train_iter/fid', fid, iter_id)

                real_images_grid = make_grid(real_images, padding=2, normalize=True)
                fake_images_grid = make_grid(fake_images, padding=2, normalize=True)

                writer.add_image('train/real_images', real_images_grid, epoch)
                writer.add_image('train/fake_images', fake_images_grid, epoch)

                iter_id += 1

            for loss_type in loss.__dict__:
                mean_loss.__dict__[loss_type] = total_loss.__dict__[loss_type] / (batch_idx + 1)
                writer.add_scalar(f'train_loss/{loss_type}', mean_loss.__dict__[loss_type], epoch)
                status_df[loss_type]['train'] = mean_loss.__dict__[loss_type]

            """mean classification accuracy for real, fake and real+fake"""
            real_acc = 100. * real_correct / real_total
            fake_acc = 100. * fake_correct / fake_total
            total_acc = 100. * (real_correct + fake_correct) / (real_total + fake_total)

            if params.train_gen_metrics:
                train_inception_score /= (batch_idx + 1)
                train_fid /= (batch_idx + 1)
                writer.add_scalar('train/fid', train_fid, epoch)
                writer.add_scalar('train/inception_score', train_inception_score, epoch)
                status_df['is']['train'] = train_inception_score
                status_df['fid']['train'] = train_fid

            writer.add_scalar('train/total_acc', total_acc, epoch)
            writer.add_scalar('train/real_acc', real_acc, epoch)
            writer.add_scalar('train/fake_acc', fake_acc, epoch)

            status_df['total_acc']['train'] = total_acc
            status_df['real_acc']['train'] = real_acc
            status_df['fake_acc']['train'] = fake_acc

            if epoch % train_params.validate_gap == 0:
                valid_loss, valid_acc, valid_gen, valid_time = evaluate(
                    all_modules, valid_dataloader, composite_loss, inception_model, fid_model, upsample, device, params)
                print('\nvalidation time: {:.3f}'.format(valid_time))

                valid_real_acc, valid_fake_acc, valid_total_acc = valid_acc
                valid_inception_score, valid_fid, valid_real_images_grid, valid_fake_images_grid, = valid_gen

                for loss_type in loss.__dict__:
                    writer.add_scalar(f'valid_loss/{loss_type}', valid_loss.__dict__[loss_type], epoch)
                    status_df[loss_type]['valid'] = valid_loss.__dict__[loss_type]

                writer.add_image('valid/real_images', valid_real_images_grid, epoch)
                writer.add_image('valid/fake_images', valid_fake_images_grid, epoch)

                writer.add_scalar('valid/total_acc', valid_total_acc, epoch)
                writer.add_scalar('valid/real_acc', valid_real_acc, epoch)
                writer.add_scalar('valid/fake_acc', valid_fake_acc, epoch)

                if params.eval_gen_metrics:
                    writer.add_scalar('valid/fid', valid_fid, epoch)
                    writer.add_scalar('valid/inception_score', valid_inception_score, epoch)

                    status_df['is']['valid'] = valid_inception_score
                    status_df['fid']['valid'] = valid_fid

                status_df['total_acc']['valid'] = valid_total_acc
                status_df['real_acc']['valid'] = valid_real_acc
                status_df['fake_acc']['valid'] = valid_fake_acc
            else:
                status_df.loc['valid', :] = 0

            save_weights, criterion = save_criteria.decide(status_df)

            print("Training results for epoch: {}:\n".format(epoch))
            print_stats(status_df)
            print()

            # Save checkpoint.
            if save_weights:
                model_dict = {
                    'shared_net': shared_net.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'generator': generator.state_dict(),
                    'classifier': classifier.state_dict(),
                    'discriminator_opt': discriminator_opt.state_dict(),
                    'generator_opt': generator_opt.state_dict(),
                    'status_df': status_df,
                    'criterion': criterion,
                    'epoch': epoch,
                    'timestamp': datetime.now().strftime("%y/%m/%d %H:%M:%S"),
                }
                weights_path = '{}.{:d}'.format(train_params.weights_path, epoch)

                print(f'Saving weights for criterion {criterion} to {weights_path}')
                torch.save(model_dict, weights_path)

    print('Testing...')
    test_loss, test_acc, test_gen, test_time = evaluate(
        all_modules, test_dataloader, composite_loss, inception_model, fid_model, upsample, device, params)

    test_real_acc, test_fake_acc, test_total_acc = test_acc
    test_inception_score, test_fid, test_real_images_grid, test_fake_images_grid = test_gen

    test_df = pd.DataFrame(
        np.zeros((1, len(eval_metrics)), dtype=np.float32),
        index=('test',),
        columns=eval_metrics,
    )
    for loss_type in test_loss.__dict__:
        test_df[loss_type]['test'] = test_loss.__dict__[loss_type]

    test_df['total_acc']['test'] = test_total_acc
    test_df['real_acc']['test'] = test_real_acc
    test_df['fake_acc']['test'] = test_fake_acc
    test_df['is']['test'] = test_inception_score
    test_df['fid']['test'] = test_fid

    print("Test Results:\n")
    print_stats(test_df)
    print()
    print('Test Time: {:.3f} sec'.format(test_time))


if __name__ == '__main__':
    main()
