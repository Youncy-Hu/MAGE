# Make It Move: Controllable Image-to-Video Generation with Text Descriptions
![Screenshot](examples/TI2V.gif)

This repository contains datasets and source code used in the CVPR'2022 paper ``Make It Move: Controllable Image-to-Video Generation with Text Descriptions".

***
## Update
- [X] We improved MAGE with a more prowerful autoencoder and a controller over VAE. The code and models of the improved version, MAGE+, have been released at [google drive](https://drive.google.com/drive/folders/1G6DrJxoAGsgAnyZhYfBTpyUC2BXZipWt?usp=sharing).
- [X] We proposed two no-reference evaluation metrics, action precision and referring expression precision, to evaluate the precision of fine-grained motions based on a captioning-and-matching method. (We chose [SwinBERT](https://github.com/microsoft/SwinBERT) as the captioning model. Please download the trained model on CATER-GENs at [google drive](https://drive.google.com/drive/folders/1YHyYPB8jcF7H3_X3VJTfptzdLBkN798Z?usp=sharing) and put it under 'metrics/swinbert_cater'.)
```bash
$ docker run --gpus all --ipc=host --rm -it --mount src=/home/user/SwinBERT/,dst=/videocap,type=bind --mount src=/home/user/,dst=/home/user/,type=bind -w /videocap linjieli222/videocap_torch1.7:fairscale bash -c "source /videocap/setup.sh && bash"
$ python metrics/swinbert_cater/eval_precision_run_caption_VidSwinBert.py --do_lower_case --do_test --eval_model_dir ./metrics/swinbert_cater/ --test_video_fname /home/results/
```
```bash
$ python eval_precision.py --data-root /home/user/datasets/CATER-GEN-v1 --gen-caption /home/user/results/catergenv1_diverse/generated_captions.json --mode ambiguous
```
***

## Dataset Generation
### Moving MNIST datasets
The scripts to generate Moving MNIST datasets are modified based on [Sync-DRAW](https://github.com/syncdraw/Sync-DRAW). You can run the following commands to generate Single Moving MNIST, Double Moving MNIST and our Modified Double Moving MNIST, respectively. 
```bash
$ python data/mnist_caption_single.py
$ python data/mnist_caption_double.py
$ python data/mnist_caption_double_modified.py
```
### CATER-GENs
#### Datasets Download
The original CATER-GEN-v1 and CATER-GEN-v2 used in our paper are provided at [link1](https://drive.google.com/drive/folders/1ICIP5qY1rTod-hTLz5zJSxlbrHrGrdt4?usp=sharing) and [link2](https://drive.google.com/drive/folders/1xJM7gNDCslpM8MJNYT1fqgiG8yyIl6ue?usp=sharing), respectively.
#### Create Your Own Datasets
Thanks to authors of [CATER](https://github.com/rohitgirdhar/CATER) and [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen) for making their code available, you can also generate your own datasets as following.

First, please generate videos and metadata according to the guideline of [CATER](https://github.com/rohitgirdhar/CATER). Please change the hyper-parameters including `min_objects, max_objects, num_frames, num_images, width, height`, and fix `CAM_MOTION = False, start_frame = 0`.
Then, you can generate text descriptions by running:
```bash
$ python data/gen_cater_text_anno.py
```

## MAGE
There are two stages training in our proposed baseline, MAGE. The first stage is to train a VQ-VAE encoder and decoder. The second stage is to train the remaining video generation model.
The trained models are provided at [google drive](https://drive.google.com/drive/folders/1G6DrJxoAGsgAnyZhYfBTpyUC2BXZipWt?usp=sharing).

### Environment
Our code has been tested on Ubuntu 18.04. Before starting, please configure your Anaconda environment by
```bash
$ conda create -n mage python=3.8
$ conda activate mage
$ pip install -r requirements.txt
```

### Stage 1. VQ-VAE Training
```bash
$ python train_vqvae.py --dataset mnist --data-root /data/data_file --output-folder ./models/vqvae_model_file
```

### Stage 2. MAGE Training
```bash
$ python main_mage.py --split train --config config/model.yaml --checkpoint-path ./models/MAGE/model_path 
```

### Sampling

```bash
$ python main_mage.py --split test --config config/model.yaml --checkpoint-path ./models/MAGE/model_path
```


## Citation
If you find this repository useful in your research then please cite
```
@InProceedings{hu2022mage,
    title={Make It Move: Controllable Image-to-Video Generation with Text Descriptions},
    author={Yaosi Hu and Chong Luo and Zhenzhong Chen},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```
