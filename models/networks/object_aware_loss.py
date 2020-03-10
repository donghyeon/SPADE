import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19


class MultiScaleDiscriminatorLossWrapper(nn.Module):
    def __init__(self, loss_module):
        super().__init__()
        self.loss_module = loss_module

    def forward(self, input, *args):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss_module(pred_i, *args)

                # TODO: is this for multi-gpu training?
                # When loss is all reduced, we don't need these lines.
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss_module(input, *args)


# This class uses torch.Tensor() to create a tensor, which is somewhat complicated.
# TODO: check multi-gpu training when tensor creation ops is used to alternate torch.Tensor()
class GANLossBase(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, input, target_is_real, *args):
        pass


class UnreducedBCELoss(GANLossBase):
    def forward(self, input, target_is_real, *args):
        target_tensor = self.get_target_tensor(input, target_is_real)
        unreduced_loss = F.binary_cross_entropy_with_logits(input, target_tensor, reduction='none')
        return unreduced_loss


class UnreducedLeastSquaresLoss(GANLossBase):
    def forward(self, input, target_is_real, *args):
        target_tensor = self.get_target_tensor(input, target_is_real)
        unreduced_loss = F.mse_loss(input, target_tensor, reduction='none')
        return unreduced_loss


class UnreducedHingeLoss(GANLossBase):
    def forward(self, input, target_is_real, for_discriminator, *args):
        if for_discriminator:
            if target_is_real:
                minval = torch.min(input - 1, self.get_zero_tensor(input))
            else:
                minval = torch.min(-input - 1, self.get_zero_tensor(input))
            unreduced_loss = -minval
        else:
            assert target_is_real, "The generator's hinge loss must be aiming for real"
            unreduced_loss = -input
        return unreduced_loss


class UnreducedWassersteinLoss(GANLossBase):
    def forward(self, input, target_is_real, *args):
        if target_is_real:
            unreduced_loss = -input
        else:
            unreduced_loss = input
        return unreduced_loss


class UnreducedLossReducer(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        if reduction not in ['mean']:
            raise ValueError('Unexpected reduction mode {}'.format(reduction))

    @staticmethod
    def reduce_mean(unreduced_loss):
        return torch.mean(unreduced_loss)

    def forward(self, unreduced_loss, *args):
        if self.reduction == 'mean':
            return self.reduce_mean(unreduced_loss)


class ObjectAwareLossReducer(nn.Module):
    def __init__(self, reduction='mean', weights=None):
        super().__init__()
        if weights is None:
            weights = [9, 1]
        self.loss_reducer = UnreducedLossReducer(reduction)
        self.weights = weights
        self.oa_mapper = WeightedObjectAwareMapper(weights)

    def forward(self, unreduced_loss, input_semantics):
        weights_map = self.oa_mapper(input_semantics)
        weights_map = F.interpolate(weights_map, unreduced_loss.shape[-2:])
        unreduced_oa_loss = weights_map * unreduced_loss
        return self.loss_reducer(unreduced_oa_loss)


class ReducedGANLoss(nn.Module):
    def __init__(self, unreduced_loss_module, reducer_module):
        super().__init__()
        self.unreduced_loss_module = unreduced_loss_module
        self.reducer_module = reducer_module

    def forward(self, input, input_semantics, *args):
        unreduced_loss = self.unreduced_loss_module(input, *args)
        loss = self.reducer_module(unreduced_loss, input_semantics)
        return loss


class ReducedL1Loss(nn.Module):
    def __init__(self, reducer_module):
        super().__init__()
        self.unreduced_loss_module = nn.L1Loss(reduction='none')
        self.reducer_module = reducer_module

    def forward(self, input, target, input_semantics):
        unreduced_loss = self.unreduced_loss_module(input, target)
        loss = self.reducer_module(unreduced_loss, input_semantics)
        return loss


# TODO: apply_oa_loss argument seems weird. Refactor this class.
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None, apply_oa_loss=False):
        super().__init__()
        if gan_mode not in ['original', 'ls', 'hinge', 'w']:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))
        self.gan_mode = gan_mode
        self.apply_oa_loss = apply_oa_loss
        self.unreduced_gan_loss_module = None
        self.reducer_module = None
        self.reduced_gan_loss_module = None
        self.multi_scale_discriminator_loss = None
        self.build(gan_mode, target_real_label, target_fake_label, tensor)

    def build(self, gan_mode, target_real_label, target_fake_label, tensor):
        if gan_mode == 'original':
            unreduced_gan_loss_module = UnreducedBCELoss(target_real_label, target_fake_label, tensor)
        elif gan_mode == 'ls':
            unreduced_gan_loss_module = UnreducedLeastSquaresLoss(target_real_label, target_fake_label, tensor)
        elif gan_mode == 'hinge':
            unreduced_gan_loss_module = UnreducedHingeLoss(target_real_label, target_fake_label, tensor)
        elif gan_mode == 'w':
            unreduced_gan_loss_module = UnreducedWassersteinLoss(target_real_label, target_fake_label, tensor)
        self.unreduced_gan_loss_module = unreduced_gan_loss_module

        if self.apply_oa_loss:
            self.reducer_module = ObjectAwareLossReducer()
        else:
            self.reducer_module = UnreducedLossReducer()

        self.reduced_gan_loss_module = ReducedGANLoss(self.unreduced_gan_loss_module, self.reducer_module)
        self.multi_scale_discriminator_loss = MultiScaleDiscriminatorLossWrapper(self.reduced_gan_loss_module)

    def forward(self, input, input_semantics, target_is_real, for_discriminator):
        loss = self.multi_scale_discriminator_loss(input, input_semantics, target_is_real, for_discriminator)
        return loss


# TODO: apply_oa_loss argument seems weird. Refactor this class.
class FeatLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor, opt=None, apply_oa_loss=False):
        super().__init__()
        self.tensor = tensor
        self.apply_oa_loss = apply_oa_loss
        self.reducer_module = None
        self.reduced_loss_module = None
        self.build()

    def build(self):
        if self.apply_oa_loss:
            self.reducer_module = ObjectAwareLossReducer()
        else:
            self.reducer_module = UnreducedLossReducer()
        self.reduced_loss_module = ReducedL1Loss(self.reducer_module)

    def forward(self, pred_fake, pred_real, input_semantics):
        num_D = len(pred_fake)
        loss = self.tensor(1).fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.reduced_loss_module(
                    pred_fake[i][j], pred_real[i][j].detach(), input_semantics)
                loss += unweighted_loss / num_D
        return loss


# TODO: apply_oa_loss argument seems weird. Refactor this class.
class VGGLoss(nn.Module):
    def __init__(self, opt=None, apply_oa_loss=False):
        super().__init__()
        self.vgg = VGG19().cuda()
        self.apply_oa_loss = apply_oa_loss
        self.reducer_module = None
        self.reduced_loss_module = None
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.build()

    def build(self):
        if self.apply_oa_loss:
            self.reducer_module = ObjectAwareLossReducer()
        else:
            self.reducer_module = UnreducedLossReducer()
        self.reduced_loss_module = ReducedL1Loss(self.reducer_module)

    def forward(self, x, y, input_semantics):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.reduced_loss_module(x_vgg[i], y_vgg[i].detach(), input_semantics)
        return loss


coco_non_object_labels = {0, 94, 97, 101, 102, 103, 106, 111, 113, 114,
                          115, 116, 117, 118, 119, 120, 124, 125, 126, 127,
                          129, 131, 132, 134, 135, 136, 138, 139, 140, 141,
                          142, 143, 145, 146, 147, 148, 149, 150, 151, 152,
                          154, 155, 157, 159, 160, 162, 164, 167, 168, 169,
                          171, 172, 173, 174, 175, 176, 177, 178, 179, 182,
                          183}  # index 183 is unknown class label
coco_num_classes_without_unknown_class = 182


class ObjectAwareMapper(nn.Module):
    def __init__(self, object_labels=None, non_object_labels=None, num_classes=None):
        super().__init__()
        if object_labels is None and non_object_labels is None and num_classes is None:
            # set to default coco dataset with unknown class
            non_object_labels = coco_non_object_labels
            num_classes = coco_num_classes_without_unknown_class + 1  # with unknown class

        if object_labels is None:
            self.non_object_indices = non_object_labels
            self.object_indices = set(range(num_classes + 1)) - self.non_object_indices
        else:
            self.object_indices = object_labels
            self.non_object_indices = set(range(num_classes + 1)) - self.object_indices

    def is_object_or_not(self, label_index):
        if label_index in self.object_indices:
            return True
        else:
            return False

    def get_object_map(self, input_semantics):
        return torch.sum(input_semantics[:, list(self.object_indices)], dim=1, keepdim=True)

    def get_non_object_map(self, input_semantics):
        return 1 - self.get_object_map(input_semantics)

    def forward(self, input_semantics):
        return self.get_object_map(input_semantics)


class WeightedObjectAwareMapper(ObjectAwareMapper):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.weights_tensor = None

    def get_weights_tensor(self):
        if self.weights_tensor is None:
            self.weights_tensor = torch.tensor(self.weights, device='cuda')
        return self.weights_tensor

    def get_weights_map(self, input_semantics):
        object_map = self.get_object_map(input_semantics)
        non_object_map = self.get_non_object_map(input_semantics)
        weights_tensor = self.get_weights_tensor()
        weights_map = weights_tensor[0] * object_map + weights_tensor[1] * non_object_map
        weights_map = weights_map / torch.sum(weights_map) * torch.sum(object_map + non_object_map)
        return weights_map

    def forward(self, input_semantics):
        return self.get_weights_map(input_semantics)
