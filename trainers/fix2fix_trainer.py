from models.fix2fix_model import Fix2FixModel
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch
from apex import amp
from apex.parallel import DistributedDataParallel


class Fix2FixTrainer(Pix2PixTrainer):
    def __init__(self, opt):
        self.opt = opt

        self.pix2pix_model = Fix2FixModel(opt)

        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr

        if opt.fp16:
            self.pix2pix_model, [self.optimizer_G, self.optimizer_D] = amp.initialize(
                self.pix2pix_model, [self.optimizer_G, self.optimizer_D], num_losses=2)
        self.generated = None

        if len(opt.gpu_ids) > 1:
            self.pix2pix_model = DistributedDataParallel(self.pix2pix_model, delay_allreduce=True)

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()

        if self.opt.fp16:
            with amp.scale_loss(g_loss, self.optimizer_G, loss_id=0) as scaled_g_loss:
                scaled_g_loss.backward()
        else:
            g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()

        if self.opt.fp16:
            with amp.scale_loss(d_loss, self.optimizer_D, loss_id=1) as scaled_d_loss:
                scaled_d_loss.backward()
        else:
            d_loss.backward()

        self.optimizer_D.step()
        self.d_losses = d_losses

    def save(self, epoch):
        self.pix2pix_model.save(epoch)
