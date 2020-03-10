import torch
from models.apex_model import ApexModel
import models.networks.object_aware_loss as oa_loss


# TODO: apply_oa_loss argument seems weird. Refactor this class.
class ObjectAwareApexModel(ApexModel):
    def set_losses(self):
        # set loss functions
        opt = self.opt
        if opt.isTrain:
            self.criterionGAN = oa_loss.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt, apply_oa_loss=True)
            if not opt.no_ganFeat_loss:
                self.criterionFeat = oa_loss.FeatLoss(tensor=self.FloatTensor, opt=self.opt, apply_oa_loss=True)
            if not opt.no_vgg_loss:
                self.criterionVGG = oa_loss.VGGLoss(opt=self.opt, apply_oa_loss=True)

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
