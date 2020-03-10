import torch
from models.pix2pix_model import Pix2PixModel
import models.networks as networks
import util.util as util


class ApexModel(Pix2PixModel):
    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.set_tensor()
        self.netG, self.netD, self.netE = self.initialize_networks(opt)
        if opt.isTrain:
            self.set_losses()

    def set_tensor(self):
        self.FloatTensor = torch.FloatTensor
        self.ByteTensor = torch.ByteTensor
        if self.use_gpu():
            self.FloatTensor = torch.cuda.FloatTensor
            self.ByteTensor = torch.cuda.ByteTensor
        if self.opt.fp16:
            self.FloatTensor = torch.cuda.HalfTensor

    def set_losses(self):
        # set loss functions
        opt = self.opt
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
