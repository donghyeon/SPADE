
import torch
from .train_options import TrainOptions
from .test_options import TestOptions


class ApexTrainOptions(TrainOptions):
    def initialize(self, parser):
        TrainOptions.initialize(self, parser)

        # for apex training
        parser.add_argument('--trainer', type=str, default='apex', help='apex|pix2pix')
        parser.add_argument('--fp16', action='store_true', help='mixed precision training')
        parser.add_argument('--local_rank', default=0, type=int, help='for DistributedDataParallel module')

        return parser

    def parse(self, save=False):
        opt = super().parse(save)

        opt.distributed = False
        if len(opt.gpu_ids) > 1:
            opt.distributed = True
            torch.cuda.set_device(opt.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

        self.opt = opt
        return opt


# TODO: Implement multi-gpu testing first
class ApexTestOptions(TestOptions):
    def initialize(self, parser):
        TestOptions.initialize(self, parser)

        # for apex training
        parser.add_argument('--local_rank', default=0, type=int, help='for DistributedDataParallel module')

        return parser
