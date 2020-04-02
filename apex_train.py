"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.apex_options import ApexTrainOptions
import torch
import data
import trainers
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from distutils.version import StrictVersion

# TODO: opt.continue_train can affect how to shuffle dataset.
#  DataLoader may need a synchronized random seed at the start of training.
#  In addition, DistributedSampler in torch 1.1 does not support shuffle argument.
def apex_create_dataloader(opt):
    dataset = data.find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))

    if StrictVersion(torch.__version__) > StrictVersion('1.1'):
        sampler = torch.utils.data.distributed.DistributedSampler(instance, shuffle=not opt.serial_batches)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(instance)

    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=int(opt.batchSize / opt.world_size),
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
        sampler=sampler
    )
    return dataloader


# parse options
opt = ApexTrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
if opt.distributed:
    dataloader = apex_create_dataloader(opt)
else:
    dataloader = data.create_dataloader(opt)

# TODO: replace vacancy's SynchronizedBatchNorm to apex.parallel.SyncBatchNorm.
#  For now, --norm_G should not contain 'syncbatch'
#  Thus, for SPADEGenerator, the --norm_G should be changed since its default is 'spectralspadesyncbatch3x3'.
# create trainer for our model
trainer = trainers.create_trainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if opt.local_rank == 0:
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                visuals = OrderedDict([('input_label', data_i['label']),
                                       ('synthesized_image', trainer.get_latest_generated()),
                                       ('real_image', data_i['image'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far,
                                                   iter_counter.epoch_iter)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
