import argparse
from losses import CDLoss, GRAMLoss, TriLoss
import os
import random
import warnings
import numpy as np

import registry
import datafree
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datafree.utils.fmix import sample_and_apply

parser = argparse.ArgumentParser(
    description='Data-free Knowledge Distillation')
# Our Implement
parser.add_argument('--teacher_weights', default=None, type=str,
                    help='teacher model weights path.')
parser.add_argument('--sampling', default='distance', type=str,
                    help='triplet feature representations for student training'
                    ', chosen from [global, features, logits].')
parser.add_argument('--triplet', default=0, type=float,
                    help='use triplet teacher for student training.')
parser.add_argument('--triplet_target', default='teacher', type=str,
                    help='use triplet teacher or student for image generation.')
parser.add_argument('--striplet', default=0, type=float,
                    help='use triplet teacher for student training.')
parser.add_argument('--striplet_feature', default='features', type=str,
                    help='triplet feature representations for student training'
                    ', chosen from [global, features, logits].')
parser.add_argument('--cd_loss', default=0, type=float,
                    help='use cd loss for student training.')
parser.add_argument('--gram_loss', default=0, type=float,
                    help='use gram loss for student training.')
parser.add_argument('--pair_sample', action='store_true',
                    help='use feat loss for student training.')
parser.add_argument('--fmix', action='store_true',
                    help='use feat loss for data mixing.')
parser.add_argument('--custom_steps', default=1.0, type=float,
                    help='custom steps for student training')
parser.add_argument('--start_layer', default=0, type=int,
                    help='start index of layers for triplet training.')
parser.add_argument('--end_layer', default=4, type=int,
                    help='end index of layers for triplet training.')
# Data Free
parser.add_argument('--adv', default=0, type=float,
                    help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float,
                    help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float,
                    help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--cr', default=0, type=float,
                    help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--cr_T', default=0, type=float,
                    help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float,
                    help='scaling factor for activation loss used in DAFL')
parser.add_argument('--save_dir', default='run/synthesis', type=str)

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float,
                    help='initial learning rate for generation')
parser.add_argument('--T', default=1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# Misc
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--balanced_sampling', action='store_true',
                    help='balanced sampling for sync loss')
parser.add_argument('--cmi', action='store_true',
                    help='balanced sampling for sync loss')
best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1
    args.gpu = gpu
    # args.gpu = None
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    args.autocast = datafree.utils.dummy_ctx

    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = '%s-%s-%s' % (args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree/log-%s-%s-%s%s.txt' % (
        args.dataset, args.teacher, args.student, args.log_tag))
    for k, v in datafree.utils.flatten_dict(vars(args)).items():
        args.logger.info("%s: %s" % (k, v))

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(
        name=args.dataset, data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model
    student = registry.get_model(args.student, num_classes=num_classes, pretrained=args.pretrained)
    teacher = registry.get_model(
        args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = datafree.utils.Normalizer(
        **registry.NORMALIZE_DICT[args.dataset])
    if args.teacher_weights is None:
        args.teacher_weights = 'checkpoints/pretrained/%s_%s.pth' % (
            args.dataset, args.teacher)
    args.logger.info('-[Init] load teacher weights from %s' %
                     args.teacher_weights)
    teacher.load_state_dict(torch.load(
        args.teacher_weights, map_location='cpu')['state_dict'])
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = datafree.criterions.KLDiv(T=args.T)

    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if 'imagenet' in args.dataset:
        img_size = 224
        nz, ngf = 512, 128
    else:
        img_size = 32
        nz, ngf = 256, 64

    generator = datafree.models.generator.Generator(
            nz=nz, ngf=ngf, img_size=img_size, nc=3)
    generator = prepare_model(generator)
    if not args.cmi:
        synthesizer = datafree.synthesis.AdvTripletSynthesizer(teacher, student, generator, args.pair_sample,
                                                        nz=nz, num_classes=num_classes, img_size=(
                                                            3, img_size, img_size), start_layer=args.start_layer, end_layer=args.end_layer,
                                                        iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                                                        synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                                                        adv=args.adv, bn=args.bn, oh=args.oh, triplet=args.triplet,
                                                        save_dir=args.save_dir, transform=ori_dataset.transform,
                                                        normalizer=args.normalizer, device=args.gpu, 
                                                        triplet_target=args.triplet_target,
                                                        balanced_sampling=args.balanced_sampling)
    else:
        feature_layers = None
        if args.teacher=='resnet34': # only use blocks
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator, 
                                                        nz=nz, num_classes=num_classes, img_size=(3, 32, 32), 
                                                        # if feature layers==None, all convolutional layers will be used by CMI.
                                                        feature_layers=feature_layers, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=None,
                                                        iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                                                        synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                                                        adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                                                        save_dir=args.save_dir, transform=ori_dataset.transform,
                                                        normalizer=args.normalizer, device=args.gpu)

    ############################################
    # Setup loss
    ############################################
    fake = torch.rand((1, 3, img_size, img_size)).cuda()
    _, _, t_layers = teacher(fake, return_features=True)
    _, _, s_layers = student(fake, return_features=True)

    gram_layers = []
    for t, s in zip(t_layers, s_layers):
        gram_layers.append(nn.Conv2d(s.size(1), t.size(1), 1).cuda())

    gram_layers = nn.ModuleList(gram_layers)
    compute_gram_loss = GRAMLoss(linears=gram_layers)
    compute_cd_loss = CDLoss(linears=gram_layers)

    ############################################
    # Setup optimizer
    ############################################
    # optimizer = torch.optim.SGD(student.parameters(
    # ), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    optimizer = torch.optim.SGD([
        {'params': student.parameters(), 'lr': args.lr},
        {'params': compute_gram_loss.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay, momentum=0.9)
    #milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    #scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try:
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))

    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):

        args.current_epoch = epoch

        for _ in range(args.ep_steps//args.kd_steps):  # total kd_steps < ep_steps
            # 1. Data synthesis
            vis_results = synthesizer.synthesize()  # g_steps
            # 2. Knowledge distillation
            train(synthesizer, [student, teacher],
                  criterion, optimizer, args, epoch, compute_cd_loss, compute_gram_loss, img_size)  # kd_steps

        # for vis_name, vis_image in vis_results.items():
        #     import IPython
        #     IPython.embed()
        #     exit()
        #     datafree.utils.save_image_batch(
        #         vis_image, 'checkpoints/datafree/%s%s.png' % (vis_name, args.log_tag))

        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                         .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr']))
        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/datafree/%s-%s-%s.pth' % (
            args.dataset, args.teacher, args.student)
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)
    args.logger.info("Best: %.4f" % best_acc1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(synthesizer, model, criterion, optimizer, args, epoch, compute_cd_loss, compute_gram_loss, img_size=32):
    loss_metric = AverageMeter()
    cd_loss_metric = AverageMeter()
    gram_loss_metric = AverageMeter()
    triplet_loss_metric = AverageMeter()
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1, 5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    compute_triplet_loss = TriLoss()

    for iter in tqdm(range(args.kd_steps)):
        images = synthesizer.sample()
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
        if args.fmix:
            images, _, _ = sample_and_apply(
                images.cpu(), alpha=1, decay_power=3, shape=(img_size, img_size))
            images = images.type(torch.FloatTensor)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat, t_layers = teacher(images, return_features=True)
            s_out, s_feat, s_layers = student(
                images.detach(), return_features=True)
            ce_loss = criterion(s_out, t_out.detach())

        if args.cd_loss and epoch < args.custom_steps * args.epochs:
            cd_loss = args.cd_loss * compute_cd_loss(s_layers, t_layers)
        else:
            cd_loss = torch.tensor(0.)

        if args.gram_loss and epoch < args.custom_steps * args.epochs:
            gram_loss = args.gram_loss * compute_gram_loss(s_layers, t_layers)
        else:
            gram_loss = torch.tensor(0.)

        if args.striplet and epoch < args.custom_steps * args.epochs:
            triplet_features = []
            if 'features' in args.striplet_feature:
                triplet_features.extend(s_layers[args.start_layer:args.end_layer])
            if 'global' in args.striplet_feature:
                triplet_features.append(s_feat)
            if 'logits' in args.striplet_feature :
                triplet_features.append(s_out)
            triplet_loss = args.striplet * compute_triplet_loss(
                triplet_features, s_out, torch.argmax(t_out, dim=-1)
            )
        else:
            triplet_loss = torch.tensor(0.)

        loss_s = ce_loss + cd_loss + gram_loss + triplet_loss
        optimizer.zero_grad()
        loss_s.backward()
        optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(ce_loss.item(), images.size(0))
        cd_loss_metric.update(cd_loss.item(), images.size(0))
        gram_loss_metric.update(gram_loss.item(), images.size(0))
        triplet_loss_metric.update(triplet_loss.item(), images.size(0))
        if args.print_freq > 0 and iter % args.print_freq == 0:
            train_acc1, train_acc5 = acc_metric.get_results()
            # args.logger.info('[Train] Epoch={current_epoch}, train_acc@1={train_acc1:.4f}, '
            #                  'train_acc@5={train_acc5:.4f}, ce_Loss={ce_loss:.4f}, cd_Loss={cd_loss:.4f}, '
            #                  'gram_Loss={gram_loss:.4f}, triplet_Loss={triplet_loss:.4f}, Lr={lr:.4f}'
            #                  .format(current_epoch=args.current_epoch, total_iters=args.kd_steps,
            #                          train_acc1=train_acc1, train_acc5=train_acc5, ce_loss=loss_metric.avg,
            #                          cd_loss=cd_loss_metric.avg, gram_loss=gram_loss_metric.avg,
            #                          triplet_loss=triplet_loss_metric.avg, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset(), cd_loss_metric.reset()
            gram_loss_metric.reset(), triplet_loss_metric.reset()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


if __name__ == '__main__':
    main()
