import torch
import pandas as pd
import torch.nn as nn

from A5_utils import REAL_UNLABELED, FAKE_UNLABELED, REAL_LABEL, FAKE_LABEL
# ***************************************************************************************************** #
#                      Documentation                        #
# ***************************************************************************************************** #
# In this Assignment, it is supposed to implement a GAN with both classifier network and #
# discriminator network. These two networks share a same network called SharedNet as input #
# The structure would be like SharedNet takes images(real/fake) as input, and outputs tensor#
# layers, where both Disriminator and Classifier take it as input to generator predictions #
# labels. The intuitive illustration would be the Shared Net handles layers before the   #
# bottleneck of the original GAN discriminator, and Discriminator and Classifier are the   #
# parts after the bottleneck. Overall, the Discriminator and Classifier take same input value#
# but optimized for different purposes. More specifically, Discriminator decide either input #
# value is from real world data or generated by Generator network, which means class size #
# is 2. Classifier would classify each input data to 20 classes, 10 for real and 10 for #
# fake. Thus, the classifier does part of the Discriminator's job. After trained these model#
# the Generator would be trained to fool the Discriminator.                   #
# Design:                                                #
#   - SharedNet:                                           #
#      3 Convolutional Layers followed by BatchNorm and LeakyReLU output => 20x20x24   #
#      1 Fully Connected layer to flatten the data for future use            #
#   - Discriminator:                                         #
#      2 Linear layers followed by Sigmoid activition function. Because Sigmoid does a #
#      greate job for binary classification                          #
#      BCELoss is being used for this model for the same reason as Sigmoid.      #
#   - Classifier:                                          #
#      2 Linear layers followed by Softmax function as it does a greate job on cla- #
#      -ssification job for multiple lables.  Output => 20 classes            #
#      CrossEntropyLoss is used here for classification purpose               #
#   - Generator:                                           #
#      4 Transposed Convolution layers are applied here. These are taken from the    #
#      tutorial given in class and A5 description to tranform the 100x1 input data to #
#      32x32x3 Cifar-10 liked dataset in order to fit the furture networks        #
#   - Optimizer:                                           #
#      All models are uses same Adam optimizer with same parameters            #
#                                                    #
# In addition for classification loss, I treat real labels and fake labels with different #
# loss function parameters. For real ones, I set ignore_index = -1 to ignore all unlabeled#
# data and ignore_index=-2 for fake datasets correspondingly.                   #
# ***************************************************************************************************** #

class TrainParams:
    """
    :ivar n_workers: no. of threads for loading data

    :ivar validate_gap: gap in epochs between validations

    :ivar tb_path: folder where to save tensorboard data

    :ivar load_weights:
        0: train from scratch,
        1: load and test
        2: load if it exists and continue training

    """

    def __init__(self):
        self.n_workers = 0
        self.batch_size = 128
        self.n_epochs = 50
        self.load_weights = 1
        self.tb_path = './tensorboard'
        #self.weights_path ="drive/My Drive/Cmput 328/Assignment5"
        
        self.weights_path = './checkpoints/model.pt'
        self.validate_gap = 10


class OptimizerParams:
    def __init__(self):
        self.type = 0
        self.lr = 0.0002
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.beta1 = 0.5


class SharedNet(nn.Module):
    class Params:
        dummy = 0

    def __init__(self, params, n_channels=3, img_size=32):
        """

        :param SharedNet.Params params:
        """
        super(SharedNet, self).__init__()
        self.conv = nn.Sequential(
            # input size 3x32x32
            nn.Conv2d(n_channels,6,kernel_size=5),
            nn.LeakyReLU(0.2),
            # input size 6x28x28
            nn.Conv2d(6,12,kernel_size = 5),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2),
            # input size 12x20x20
            nn.Conv2d(12,24,kernel_size=5),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2)
            # output size 24x20x20
        )

        self.fc = nn.Sequential(
            nn.Linear(24*20*20,1024),
            nn.ReLU()
        )

        #pass

    def init_weights(self):
        def init_weight(m):
          cln = m.__class__.__name__
          if cln.find('Conv')!=-1:
            nn.init.normal_(m.weight.data,0.0,0.2)
          elif cln.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)
        self.conv.apply(init_weight)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x


class Discriminator(nn.Module):
    class Params:
        opt = OptimizerParams()

    def __init__(self, params, n_channels=3, img_size=32):
        """

        :param Discriminator.Params params:
        """
        super(Discriminator, self).__init__()
        #pass
        self.conv = nn.Sequential(
            # intput size 24x20x20
            nn.Linear(1024,256), 
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param enc_input:
        :return:
        """
        #pass

        #x = x.view(x.shape[0],-1)
        output = self.conv(x)
        output = output.reshape(-1)
        return output

    def get_loss(self, dscr_out, labels):
        """

        :param dscr_out: Discriminator output
        :param labels: Real vs fake binary labels (real --> 1, fake --> 0)
        :return:
        """
        #pass
        #print('size of dscr_out is',len(dscr_out),'size of lables is',len(labels),'\n')
        #print(dscr_out,'\t','labels')
        
        loss = nn.BCELoss()
        return loss(dscr_out,labels)

    def get_optimizer(self, modules):
        """

        :param nn.ModuleList modules: [shared_net, discriminator, classifier, generator, composite_loss]
        :return:
        """
        opt = OptimizerParams()
        opt_params = (list(modules[0].parameters())+list(modules[1].parameters())+list(modules[2].parameters()))
        
        return torch.optim.Adam(opt_params, lr=opt.lr, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

    def init_weights(self):
        pass


class Classifier(nn.Module):
    class Params:
        dummy = 0

    def __init__(self, params, n_classes=20, n_channels=3, img_size=32):
        """

        :param n_classes: 10 classes for real images and 10 for fake images
        :param Classifier.Params params:
        """
        super(Classifier, self).__init__()
        #pass
        self.net = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,20),
            nn.Softmax()
        )

    def get_loss(self, cls_out, labels, is_labeled=None, n_labeled=None):
        """

        :param cls_out: Classifier output
        :param labels: labels for both fake and real images in range 0 - 19;
        -1 for real unlabeled, -2 for fake unlabeled
        :param is_labeled: boolean array marking which labels are valid; None when all are valid
        :param n_labeled: number of valid labels;  None when all are valid
        :return:
        """
        #pass

        if n_labeled and n_labeled == 0:
          return torch.tensor(0.,requires_grad=True)

        loss_real = nn.CrossEntropyLoss(ignore_index=-1)
        loss_fake = nn.CrossEntropyLoss(ignore_index=-2)
        #temp = torch.tensor(0.,requires_grad=True)

        """for i in range(len(is_labeled)):
          if is_labeled[i] >= 0:
            print(cls_out[i].item,labels[i])
            temp += loss(cls_out[i],labels[i])"""

        if -1 in labels:
          #print('used -1 real')
          return loss_real(cls_out,labels)
        else:
          #print('used -2 fake')
          #print(loss_fake(cls_out,labels))
          return loss_fake(cls_out,labels)
        
        #return loss()
    def init_weights(self):
        pass

    def forward(self, x):
        #pass
        #x = x.view(x.shape[0],-1)
        x = self.net(x)
        #x = x.reshape(-1)
        return x

class Generator(nn.Module):
    class Params:
        opt = OptimizerParams()

    def __init__(self, params, input_size, n_channels=3, out_size=32):
        """

        :param Generator.Params params:
        """
        super(Generator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100,256,4,1,0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
      x = self.conv(x)
      #print("\n output of generator is",x.shape,"\n")
      return x

    def get_optimizer(self, module):
        """

        :param nn.ModuleList modules: [shared_net, discriminator, classifier, generator, composite_loss]
        :return:
        """
        opt = OptimizerParams()
        return torch.optim.Adam(module.parameters(),lr=opt.lr, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

    def init_weights(self):
        def init_weight(m):
          cln = m.__class__.__name__
          if cln.find('Conv')!=-1:
            nn.init.normal_(m.weight.data,0.0,0.2)
          elif cln.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data,1.0,0.02)
            nn.init.constant_(m.bias.data,0)
        self.conv.apply(init_weight)


class CompositeLoss(nn.Module):
    class Params:
        dummy = 0

    def __init__(self, device, params):
        """

        :param torch.device device:
        :param CompositeLoss.Params params:
        """
        super(CompositeLoss, self).__init__()

    def forward(self, dscr_loss, cls_loss):
        #pass
        x = dscr_loss+cls_loss

        return x

class SaveCriteria:
    def __init__(self, status_df):
        """

        :param pd.DataFrame status_df:
        """
        self._opt_status_df = status_df.copy()

    def decide(self, status_df):
        """
        decide when to save new checkpoint while training based on training and validation stats
        following metrics are available:
      |   dscr_real |   cls_real |   cmp_real |   dscr_fake |   cls_fake |   cmp_fake |   cmp |   dscr_gen |   cls_gen |
         cmp_gen |   total_acc |   real_acc |   fake_acc |   fid |   is |

         where the first 10 are losses: dscr --> discriminator, cls --> classifier, gen --> generator,
         cmp --> composite, real --> real images, fake --> fake images

         acc --> classification accuracy
         is --> inception_score

        :param pd.DataFrame status_df:
        """

        save_weights = 0
        criterion = ''

        """total train accuracy over real+fake images"""
        if status_df['total_acc']['valid'] > self._opt_status_df['total_acc']['valid']:
            self._opt_status_df['total_acc']['valid'] = status_df['total_acc']['valid']
            save_weights = 1
            criterion = 'valid_acc'

        if status_df['total_acc']['train'] > self._opt_status_df['total_acc']['train']:
            self._opt_status_df['total_acc']['train'] = status_df['total_acc']['train']
            save_weights = 1
            criterion = 'train_acc'

        """composite loss on real images"""
        if status_df['cmp_real']['valid'] > self._opt_status_df['cmp_real']['valid']:
            self._opt_status_df['cmp_real']['valid'] = status_df['cmp_real']['valid']
            save_weights = 1
            criterion = 'valid_loss'

        if status_df['cmp_real']['train'] > self._opt_status_df['cmp_real']['train']:
            self._opt_status_df['cmp_real']['train'] = status_df['cmp_real']['train']
            save_weights = 1
            criterion = 'train_loss'

        return save_weights, criterion