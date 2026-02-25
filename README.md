# Wildfire Spread Prediction with SwinUnet

This repository reproduces the experiments from:

> **Predicting Next-Day Wildfire Spread with Time Series and Attention**  
> Saad Lahrichi, Jesse Johnson, Jordan Malof  
> University of Missouri & University of Montana, 2025  
> [https://arxiv.org/abs/2502.12003](https://arxiv.org/abs/2502.12003)

## Goal

The goal of this project is to reproduce the SwinUnet-based wildfire spread prediction results reported in Lahrichi et al. (2025). The paper adapts the [SwinUnet](https://arxiv.org/abs/2105.05537) model — originally developed for medical image segmentation — to the problem of next-day wildfire spread prediction using the WildfireSpreadTS (WFTS) benchmark dataset.

Specifically, we aim to reproduce:
- SwinUnet trained from scratch and with ImageNet pre-trained weights
- Both single-day (1-day input) and multi-day (5-day input) prediction scenarios
- 12-fold leave-one-year-out cross-validation on the WFTS dataset
- Average Precision (AP) as the evaluation metric

The paper demonstrates that SwinUnet with ImageNet pre-training achieves state-of-the-art performance on the WFTS benchmark, surpassing prior convolutional and attention-based baselines (Res18-Unet, Res50-Unet, UTAE).

## Dataset

We use the **WildfireSpreadTS (WFTS)** dataset:

> Sebastian Gerard, Yu Zhao, and Josephine Sullivan.  
> *WildfireSpreadTS: A Dataset of Multi-Modal Time Series for Wildfire Spread Prediction.*  
> Advances in Neural Information Processing Systems (NeurIPS), 36:74515–74529, 2023.

The dataset includes 607 wildfire events across the western United States (2018–2021), totaling 13,607 daily multi-channel images across 23 channels (active fires, weather, topography, and vegetation) at 375m resolution.

See `notebooks/download_and_convert_dataset.ipynb` for instructions on downloading and preparing the dataset.

## 1. Download pre-trained Swin Transformer weights (Swin-T)

Pre-trained ImageNet weights (`swin_tiny_patch4_window7_224`) can be loaded from HuggingFace or downloaded manually:

- [Google Drive (original SwinUnet weights)](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): place into `pretrained_ckpt/`

## 2. Prepare data

Download and convert the WFTS dataset using the provided notebook:

```bash
jupyter notebook notebooks/download_and_convert_dataset.ipynb
```

## 3. Environment

```bash
pip install -r requirements.txt
```

Python 3.7+ is recommended.

## 4. Train / Test

Train on the wildfire dataset:

```bash
python train.py --dataset wildfire --cfg configs/swin_tiny_patch4_window4_128_wildfire.yaml \
  --root_path your/DATA_DIR --max_epochs 150 --output_dir your/OUT_DIR \
  --img_size 128 --base_lr 0.05 --batch_size 24
```

Test:

```bash
python test.py --dataset wildfire --cfg configs/swin_tiny_patch4_window4_128_wildfire.yaml \
  --volume_path your/DATA_DIR --output_dir your/OUT_DIR \
  --max_epoch 150 --base_lr 0.05 --img_size 128 --batch_size 24
```

## Results (from paper)

| Model        | Input days | Test AP       | # Params |
|--------------|------------|---------------|----------|
| SwinUnet*    | 1          | 0.383 ± 0.087 | 27.2M    |
| SwinUnet     | 1          | 0.351 ± 0.087 | 27.2M    |
| SwinUnet*    | 5          | 0.404 ± 0.081 | 27.3M    |
| SwinUnet     | 5          | 0.365 ± 0.085 | 27.2M    |
| Res18-Unet   | 5          | 0.335 ± 0.070 | 14.4M    |
| UTAE**       | 5          | 0.381 ± 0.103 | 1.1M     |

\* ImageNet pre-trained  \*\* PASTIS pre-trained

## References

- [SwinUnet (original)](https://github.com/HuCaoFighting/Swin-Unet)
- [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
- [WildfireSpreadTS dataset](https://github.com/SebastianGer/WildfireSpreadTS)

## Citation

If you use this code, please cite the following works:

```bibtex
@article{lahrichi2025wildfire,
  title={Predicting Next-Day Wildfire Spread with Time Series and Attention},
  author={Lahrichi, Saad and Johnson, Jesse and Malof, Jordan},
  journal={arXiv preprint arXiv:2502.12003},
  year={2025}
}

@article{gerard2023wildfirespreadts,
  title={{WildfireSpreadTS}: A Dataset of Multi-Modal Time Series for Wildfire Spread Prediction},
  author={Gerard, Sebastian and Zhao, Yu and Sullivan, Josephine},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={74515--74529},
  year={2023}
}

@InProceedings{swinunet,
  author    = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
  title     = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
  booktitle = {Proceedings of the European Conference on Computer Vision Workshops (ECCVW)},
  year      = {2022}
}
```
