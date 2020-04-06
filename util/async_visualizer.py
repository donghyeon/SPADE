from .visualizer import Visualizer
from collections import OrderedDict
import torch


def all_reduce_dict(dict_to_reduce):
    for key in dict_to_reduce:
        torch.distributed.all_reduce(dict_to_reduce[key], async_op=True)
    return dict_to_reduce


def all_gather_dict(dict_to_gather):
    gathered_dict = OrderedDict()
    for key in dict_to_gather:
        tensor_to_gather = dict_to_gather[key]
        gathered_tensors = [torch.ones_like(tensor_to_gather) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensors, tensor_to_gather, async_op=True)
        gathered_dict[key] = gathered_tensors
    return gathered_dict


class AsyncVisualizer(Visualizer):
    def __init__(self, opt):
        super().__init__(opt)
        self.current_epoch = None
        self.current_epoch_iter = None

        # Store previous visualizer function arguments
        self.print_current_errors_fn_args = None
        self.plot_current_errors_fn_args = None
        self.display_current_results_fn_args = None

    def print_current_errors(self, epoch, i, errors, t):
        if not self.current_epoch:
            self.store_print_fn_args(epoch, i, errors, t)
            return

        if self.need_synchronize(epoch, i):
            torch.distributed.barrier()

        old_epoch, old_i, old_errors, old_t = self.get_print_fn_args()
        super().print_current_errors(old_epoch, old_i, old_errors, old_t)
        self.store_print_fn_args(epoch, i, errors, t)

    def plot_current_errors(self, errors, step):
        if not self.current_epoch:
            self.store_plot_fn_args(errors, step)
            return

        old_errors, old_step = self.get_plot_fn_args()
        super().plot_current_errors(old_errors, old_step)
        self.store_plot_fn_args(errors, step)

    def display_current_results(self, visuals, epoch, step, iter):
        if not self.current_epoch:
            self.store_display_fn_args(visuals, epoch, step, iter)
            return

        if self.need_synchronize(epoch, iter):
            torch.distributed.barrier()

        old_visuals, old_epoch, old_step, old_iter = self.get_display_fn_args()
        super().display_current_results(old_visuals, old_epoch, old_step, old_iter)
        self.store_display_fn_args(visuals, epoch, step, iter)

    def get_print_fn_args(self):
        epoch, i, errors, t = self.print_current_errors_fn_args
        # All reduce mean by taking an average
        for key in errors:
            errors[key] /= self.opt.world_size
        return epoch, i, errors, t

    def get_plot_fn_args(self):
        # errors are already synchronized since this function is followed by print_current_errors
        errors, step = self.plot_current_errors_fn_args
        # All reduce mean by taking an average
        for key in errors:
            errors[key] /= self.opt.world_size
        return errors, step

    def get_display_fn_args(self):
        visuals, epoch, step, iter = self.display_current_results_fn_args
        # Combine the gathered outputs from other GPUs through the batch dimension
        for key in visuals:
            visuals[key] = torch.cat(visuals[key])
        return visuals, epoch, step, iter

    def store_print_fn_args(self, epoch, i, errors, t):
        errors = all_reduce_dict(errors)
        self.print_current_errors_fn_args = (epoch, i, errors, t)

    def store_plot_fn_args(self, errors, step):
        self.plot_current_errors_fn_args = (errors, step)

    def store_display_fn_args(self, visuals, epoch, step, iter):
        self.display_current_results_fn_args = (visuals, epoch, step, iter)

    def need_synchronize(self, epoch, epoch_iter):
        if self.current_epoch == epoch and self.current_epoch_iter == epoch_iter:
            return False
        else:
            self.current_epoch = epoch
            self.current_epoch_iter = epoch_iter
            return True

    def make_last_visualizations(self):
        self.print_current_errors(self.current_epoch + 1, self.current_epoch_iter + 1, None, None)
        self.plot_current_errors(None, None)
        self.display_current_results(None, self.current_epoch + 1, None, self.current_epoch_iter + 1)
