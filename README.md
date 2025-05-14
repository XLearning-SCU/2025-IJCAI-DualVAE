# DualVAE

The official repos for "**Learning Robust Multi-view Representation Using Dual-masked VAEs**" (DualVAE)

- Accepted by **IJCAI 2025**

## Abstract

Most existing multi-view representation learning methods assume view-completeness and noise-free data. However, such assumptions are strong in real-world applications. Although individual methods have been tailored to suit view-missing or noise problems, there is no one-size-fits-all approach to addressing both cases together. To this end, we propose a holistic method, called Dual-masked Variational Autoencoders (DualVAE), which aims at learning robust multi-view representation. The DualVAE exhibits an innovative amalgamation of dual-masked prediction, mixture-of-experts learning, representation disentangling, and a joint loss function in wrapping up all components. The key novelty lies in the dual-masked (view-mask and patch-mask) mechanism to mimic missing view and noisy data. Extensive experiments on four multi-view datasets show the effectiveness of the proposed method and its superior performance in comparison to baselines.

![pipeline](assets/pipeline.png)



## Data preparation

Before running experiments for 4 datasets, you can find raw datasets: Google Drive

All datasets should be organized as:

```
DualVAE 
├───MyData
│   └───coil-20
│   └───coil-100
│   └───EdgeMNISTDataset
│   └───PolyMNIST
├───configs
...
```

## Training step

First switch to the root directory of the project, then run the command :

```pyth
python train.py -f [config file]
```

After trained, a new weight file can be found in directory  `/experiments`

## Evaluation step

First fill in the path of pretrained model to `eval.model_path` domain in corresponding config file.

Then run the command:

```
python eval.py -f [config file]
```

## Environment

- pytorch == 2.3.1
- torchvision == 0.18.1
- torchinfo == 1.8.0
- yacs == 0.1.8
- scikit-learn == 1.3.2
- scipy == 1.10.1
- pillow == 10.4.0
- munkres == 1.1.4

## Cite

