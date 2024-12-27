## News
* `2024/12/20` We release the code for the *data-free knowledge distillation* tasks.

# RGAL

This is a PyTorch implementation of the following paper:

**Relation-Guided Adversarial Learning for Data-Free Knowledge Transfer**, IJCV 2024.

Yingping Liang and Ying Fu

[Paper](https://link.springer.com/article/10.1007/s11263-024-02303-4)

<img src="misc/framework.png" width="100%" >

**Abstract**: *Data-free knowledge distillation transfers knowledge by recovering training data from a pre-trained model. Despite the recent success of seeking global data diversity, the diversity within each class and the similarity among different classes are largely overlooked, resulting in data homogeneity and limited performance. In this paper, we introduce a novel Relation-Guided Adversarial Learning method with triplet losses, which solves the homogeneity problem from two aspects. To be specific, our method aims to promote both intra-class diversity and inter-class confusion of the generated samples. To this end, we design two phases, an image synthesis phase and a student training phase. In the image synthesis phase, we construct an optimization process to push away samples with the same labels and pull close samples with different labels, leading to intra-class diversity and inter-class confusion, respectively. Then, in the student training phase, we perform an opposite optimization, which adversarially attempts to reduce the distance of samples of the same classes and enlarge the distance of samples of different classes. To mitigate the conflict of seeking high global diversity and keeping inter-class confusing, we propose a focal weighted sampling strategy by selecting the negative in the triplets unevenly within a finite range of distance. RGAL shows significant improvement over previous state-of-the-art methods in accuracy and data efficiency. Besides, RGAL can be inserted into state-of-the-art methods on various data-free knowledge transfer applications. Experiments on various benchmarks demonstrate the effectiveness and generalizability of our proposed method on various tasks, specially data-free knowledge distillation, data-free quantization, and non-exemplar incremental learning.*




https://github.com/user-attachments/assets/eb78306f-1fbe-465a-9996-7315716f0b55





## Instillation

```
conda create -n rgal python=3.9
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install scipy tqdm pillow kornia 
```

## Run

The dataset (CIFAR-10/-100) will be downloaded automatically when running.

We provide a running script:
```
python main.py \
--epochs 200 \
--dataset cifar10 \
--batch_size 128 \
--synthesis_batch_size 256 \
--teacher wrn40_2 \
--student wrn16_1 \
--lr 0.1 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 400 \
--lr_g 1e-3 \
--adv 1.0 \
--bn 1.0 \
--oh 1.0 \
--act 0.001 \
--gpu 0 \
--seed 0 \
--T 20 \
--save_dir run/scratch1 \
--log_tag scratch1 \
--cd_loss 0.1 \
--gram_loss 0 \
--teacher_weights best_model/cifar10_wrn40_2.pth \
--custom_steps 1.0 \
--print_freq 50 \
--triplet_target student \
--pair_sample \
--striplet_feature global \
--start_layer 2 \
--triplet 0.1 \
--striplet 0.1 \
--balanced_sampling \
--balance 0.1
```

where "--triplet" and "--striplet" indicates the loss weights of our proposed in the data generation stage and distillation stage, separately.

To running our method on different teacher and student models, modify "--teacher" and "--student wrn16_1"

"--balanced_sampling" indicates the paired sampling strategy as in our paper.

Pretrained checkpoints for examples are available at (best_model)[https://github.com/Sharpiless/RGAL/tree/main/best_model].

![image](https://github.com/user-attachments/assets/3c8b7698-7f11-430c-ac6d-d7d0b4a22a7f)


## Visualization

Please refer to (ZSKT)[https://github.com/polo5/ZeroShotKnowledgeTransfer].

## License and Citation
This repository can only be used for personal/research/non-commercial purposes.
Please cite the following paper if this model helps your research:

```
@article{liang2024relation,
  title={Relation-Guided Adversarial Learning for Data-Free Knowledge Transfer},
  author={Liang, Yingping and Fu, Ying},
  journal={International Journal of Computer Vision},
  pages={1--18},
  year={2024},
  publisher={Springer}
}
```

## Acknowledgments
* The code for inference and training is heavily borrowed from [CMI](https://github.com/zju-vipa/CMI), we thank the author for their great effort.
