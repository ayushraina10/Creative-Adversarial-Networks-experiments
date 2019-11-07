from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


cudnn.benchmark = True
class opt:
    def __init__(self):
        self.cuda=False
        self.ngpu = 2
        self.nz = 100
        self.nc = 1
        self.ngf = 64
        self.ndf = 64
        self.netG = "blah"
        self.netD = "blahblah"
        self.batchSize = 32
        self.dataset = ""
        self.workers = 2
        self.dataroot = ""
        self.imageSize=64
        self.outf = "fake_img_for_tsne"

opt.cuda = True
opt.ngpu = 2
opt.nz = 100
opt.ngf = 64
opt.ndf = 64
opt.dataset = "wikiart"
opt.dataroot = "./datasets/wikiart"
opt.workers = 2
opt.outf = "fake_img_for_tsne"

opt.netG  = "./can_wikiart_chkpt_halflr_100_epochs/netG_epoch_99.pth"
opt.netD  = "./can_wikiart_chkpt_halflr_100_epochs/netD_epoch_99.pth"

opt.batchSize = 32
opt.imageSize = 64


device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

#opt.dataset = 'mnist'
if opt.dataset in ['imagenet', 'folder', 'lfw', 'wikiart']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    opt.nc=3

if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               #transforms.FiveCrop(opt.imageSize*(0.9)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    opt.nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=False,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        opt.nc=1


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))

class Discriminator(nn.Module):
    def __init__(self, ngpu, n_class):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.n_class = n_class
        nc = opt.nc
        ndf = opt.ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.disc = nn.Sequential(
            nn.Linear(ndf*8*4*4,1),
            nn.Sigmoid()
        )
        self.clas = nn.Sequential(
            nn.Linear((ndf*8)*4*4, 1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512, n_class)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            dis = nn.parallel.data_parallel(self.disc, output.view(-1,ndf*8*4*4), range(self.ngpu))
            cla = nn.parallel.data_parallel(self.clas,output.view(-1,ndf*8*4*4),range(self.ngpu))
        else:
            output = self.main(input)
            dis = self.disc(output.view(-1,ndf*8*4*4))
            cla = self.clas(output.view(-1,ndf*8*4*4))
        return dis.view(-1, 1).squeeze(1), cla#.view(self.n_class,1)
    def forward_tsne(self, input):        
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            tsne_out = nn.parallel.data_parallel(self.clas[0],output.view(-1,ndf*8*4*4),range(self.ngpu))
            tsne_out = nn.parallel.data_parallel(self.clas[1],tsne_out,range(self.ngpu))
            tsne_out = nn.parallel.data_parallel(self.clas[2],tsne_out,range(self.ngpu))
        else:
            output = self.main(input)
            tsne_out = self.clas[0](output.view(-1,ndf*8*4*4))
            tsne_out = self.clas[1](tsne_out)
            tsne_out = self.clas[1](tsne_out)
        return tsne_out    


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        nz =opt.nz
        nc = opt.nc
        ngf = opt.ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

n_class = 27

netG = Generator(ngpu).to(device)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)
netD = Discriminator(ngpu, n_class).to(device)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

#batch_size = 1000
#noise = torch.randn(batch_size, nz, 1, 1, device=device)
#fake = netG(noise)

        
for i,data in enumerate(dataloader,0):
        real_cpu = data[0].to(device)
        raw_label =data[1].to(device)
        batch_size = real_cpu.size(0)
        lat_im_rep = netD.forward_tsne(real_cpu).detach().to('cpu')
        raw_labels_to_concat = raw_label.float().view(-1,1).detach().to('cpu')
        
        if i==0:
           tsne_output = torch.cat((lat_im_rep,raw_labels_to_concat),1).detach().to('cpu')
        else:
           tsne_output = torch.cat((tsne_output, torch.cat((lat_im_rep,raw_labels_to_concat), 1)),0).detach().to('cpu')
        print(tsne_output.size())
        #if(i==49):
        #    break
torch.save(tsne_output, "real_img_latent_rep.pth")

for n in range(200):
    batch_size = 36
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise)
    if n==0:
       lat_im_rep = netD.forward_tsne(fake).detach().to('cpu')
    else: 
       lat_im_rep = torch.cat((lat_im_rep, netD.forward_tsne(fake).detach().to('cpu')), 0) 
    print(lat_im_rep.size())  
    vutils.save_image(fake.detach(),'%s/fake_samples_eval_n_%03d.png' % (opt.outf, n), normalize=True)


#print(lat_im_rep)



torch.save(lat_im_rep, "fake_img_latent_rep.pth")

#fake_tsne_output = netD.forward_tsne(fake)

        #label = torch.full((batch_size,), real_label, device=device)



#vutils.save_image(fake.detach(), "eval_generated_images.png", normalize=True)
