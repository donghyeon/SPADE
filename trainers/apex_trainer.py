import models
from trainers.pix2pix_trainer import Pix2PixTrainer
import torch
from apex import amp
from apex.parallel import DistributedDataParallel


class ApexTrainer(Pix2PixTrainer):
    def __init__(self, opt):
        self.opt = opt

        self.pix2pix_model = models.create_model(opt)

        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr

        if opt.fp16:
            self.pix2pix_model, [self.optimizer_G, self.optimizer_D] = amp.initialize(
                self.pix2pix_model, [self.optimizer_G, self.optimizer_D], num_losses=2)
        self.generated = None

        if opt.distributed:
            self.pix2pix_model = DistributedDataParallel(self.pix2pix_model, delay_allreduce=True)

        if opt.continue_train:
            self.load_checkpoint('latest')

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

    # To keep compatibility of *train.py, epoch argument is served for the filename
    # epoch: # of epoch or 'latest'
    # this will not call util.save_network()
    def save(self, epoch):
        filename = self.get_checkpoint_filename(epoch)
        # Only one process will save the model to disk
        if self.opt.local_rank == 0:
            self.save_checkpoint(filename)

    # For now, there's no need to save netG, netD and netE separately
    def save_checkpoint(self, filename):
        checkpoint = {'model': self.pix2pix_model.state_dict(),
                      'optimizer_G': self.optimizer_G.state_dict(),
                      'optimizer_D': self.optimizer_D.state_dict(),
                      'amp': amp.state_dict()}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, epoch):
        filename = self.get_checkpoint_filename(epoch)
        checkpoint = torch.load(filename, map_location=torch.device('cuda', torch.cuda.current_device()))
        self.pix2pix_model.load_state_dict(checkpoint['model'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        amp.load_state_dict(checkpoint['amp'])

    def get_checkpoint_filename(self, epoch):
        if epoch == 'latest':
            return 'model_latest.pth'
        else:
            return 'model_epoch_%s.pth' % epoch
