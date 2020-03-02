import torch.nn as nn
from models.networks.base_network import BaseNetwork
from models.networks.se_architecture import SPADEResnetSEBlock
from models.networks.generator import SPADEGenerator


class BaseSPADEGenerator(SPADEGenerator):
    def __init__(self, opt, base_spade_block):
        BaseNetwork.__init__(self)
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = base_spade_block(16 * nf, 16 * nf, opt)

        self.G_middle_0 = base_spade_block(16 * nf, 16 * nf, opt)
        self.G_middle_1 = base_spade_block(16 * nf, 16 * nf, opt)

        self.up_0 = base_spade_block(16 * nf, 8 * nf, opt)
        self.up_1 = base_spade_block(8 * nf, 4 * nf, opt)
        self.up_2 = base_spade_block(4 * nf, 2 * nf, opt)
        self.up_3 = base_spade_block(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = base_spade_block(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


class SPADESEGenerator(BaseSPADEGenerator):
    def __init__(self, opt):
        super().__init__(opt, SPADEResnetSEBlock)
