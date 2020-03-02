import torch
import models.networks as networks
import util.util as util
from models.pix2pix_model import Pix2PixModel


class Fix2FixModel(Pix2PixModel):
    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            if not opt.no_ganFeat_loss:
                self.criterionFeat = networks.FeatLoss(tensor=self.FloatTensor, opt=self.opt)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(opt=self.opt)

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, input_semantics, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            G_losses['GAN_Feat'] = self.criterionFeat(pred_fake, pred_real, input_semantics) \
                                   * self.opt.lambda_feat

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image, input_semantics) \
                              * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, input_semantics, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, input_semantics, True,
                                               for_discriminator=True)

        return D_losses
