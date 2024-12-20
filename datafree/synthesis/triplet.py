import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import TriLoss

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv
from datafree.utils import ImagePool, DataIter
from torchvision import transforms
from kornia import augmentation
from tqdm import tqdm


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class AdvTripletSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, pair_sample, nz, num_classes, img_size,
                 start_layer, end_layer, iterations=100, lr_g=0.1, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1, triplet=0.0,
                 save_dir='run/cmi', transform=None,
                 normalizer=None, device='cpu', distributed=False, 
                 triplet_target='teacher', balanced_sampling=False):
        super(AdvTripletSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.iterations = iterations
        self.lr_g = lr_g
        self.progressive_scale = progressive_scale
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.triplet = triplet
        self.compute_triplet_loss = TriLoss(balanced_sampling=balanced_sampling)
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.triplet_target = triplet_target

        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device).train()
        # local and global bank

        self.device = device
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m))

        self.aug = transforms.Compose([
            augmentation.RandomCrop(
                size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])
        self.pair_sample = pair_sample

    def synthesize(self, targets=None):
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        #inputs = torch.randn( size=(self.synthesis_batch_size, *self.img_size), device=self.device ).requires_grad_()
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz),
                        device=self.device).requires_grad_()
        if targets is None:
            targets = torch.randint(
                low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0]  # sort for better visualization
        targets = targets.to(self.device)

        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {
                                     'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=0.1*self.lr)
        for _ in tqdm(range(self.iterations)):
            inputs = self.generator(z)
            global_view = self.aug(inputs)  # crop and normalize

            #############################################
            # Inversion Loss
            #############################################
            t_out, _, t_layers = self.teacher(
                global_view, return_features=True)
            s_out, _, s_layers = self.student(
                global_view, return_features=True)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, targets)

            if self.adv > 0:
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                # decision adversarial distillation
                loss_adv = - \
                    (kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
            else:
                loss_adv = loss_oh.new_zeros(1)

            if self.triplet_target == 'teacher':
                triplet_layers = t_layers
            elif self.triplet_target == 'student':
                triplet_layers = s_layers
            else:
                raise NotImplementedError()

            if self.triplet > 0:
                loss_tri = self.compute_triplet_loss(
                    triplet_layers[self.start_layer:self.end_layer], t_out, torch.argmax(t_out, dim=-1))
            else:
                loss_tri = loss_oh.new_zeros(1)

            loss = self.bn * loss_bn + self.oh * loss_oh + \
                self.adv * loss_adv + self.triplet * loss_tri

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.student.train()
        self.data_pool.add(best_inputs)

        dst = self.data_pool.get_dataset(
            transform=self.transform, pair_sample=self.pair_sample)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(
                train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        print("sample_batch_size:", self.sample_batch_size)
        return {"synthetic": best_inputs, "targets": targets}

    def sample(self):
        return self.data_iter.next()
