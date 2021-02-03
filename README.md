## More Classifiers, Less Forgetting: A Generic Multi-classifier Paradigm for Incremental Learning (ECCV 2020)
- Exploit the classifier ensemble for reducing forgetting on learning tasks incrementally.
- Extend two regularization methods (MAS and LwF) focusing on parameter and activation regularization.
- Obtain consistent improvements over the single-classifier paradigm.

![architecture](https://github.com/Liuy8/MUC/blob/master/MUC_overview.png)

## Dependencies

- PyTorch 
- Python 
- Numpy
- scipy

## Data

- Download the dataset (CIFAR-100, SVHN) and save them to the 'data' directory.
- SVHN is used as an out-of-distribution dataset for training additional side classifiers.


## Experiment on CIFAR-100 incremental benchmark
- Run ```1.cifar100_MUC_LwF_phase1.py``` to train the MUC-LwF method.
  - Group Dataset 구성하는 부분을 새로운 Dataset을 생성하는 방법으로 구현함
    - 기존방법으로 하면 order를 따르는것이 아닌 새로운 학습 데이터셋을 매번 생성하는 현상이 발생함
    - 새로운 데이터셋을 생성하는 방법(torch.utils.data.TensorDataset)과 torchvision dataloader의 데이터 순서가 달라서 resnet model을 변경함(x.permute(0,3,1,2)
  - lr_scheduler.MultistepLR 이 잘 작동하지 않는것으로 보임 (정해진 milestone이 아닌 곳에서 업데이트됨)
- compute_accuracy.py : compute_accuracy_WI 함수를 사용해서 계산 -> 변경필요할수도 있음

## Experiment on Tiny-ImageNet incremental benchmark

- Run ```tinyimagenet_MUC_LwF.py``` to train the MUC-LwF method.
- 아직 못건드림 아마도 위에꺼에서 데이터셋 바꾸는게 빠를듯.

## Notes
- Some codes are based on the codebase of the [repository](https://github.com/hshustc/CVPR19_Incremental_Learning).
- More instructions will be provided later.

# Citation
Please cite the following paper if it is helpful for your research:
```
@InProceedings{MUC_ECCV2020,
author = {Liu, Yu and Parisot, Sarah and Slabaugh, Gregory and Jia, Xu and Leonardis,Ales and Tuytelaars, Tinne}
title = {More Classifiers, Less Forgetting: A Generic Multi-classifier Paradigm for Incremental Learning},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```
