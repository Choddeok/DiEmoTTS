# DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech <br><sub>The official implementation of DiEmo-TTS (INTERSPEECH 2025)</sub>

## [Paper 📄]()|[Demo 🎧](https://choddeok.github.io/DiEmo-TTS-Demo/)

**Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Seong-Whan Lee**

## Abstract
Cross-speaker emotion transfer in speech synthesis relies on extracting speaker-independent emotion embeddings for accurate emotion modeling without retaining speaker traits. However, existing timbre compression methods fail to fully separate speaker and emotion characteristics, causing speaker leakage and degraded synthesis quality. To address this, we propose DiEmo-TTS, a self-supervised distillation method to minimize emotional information loss and preserve speaker identity. We introduce cluster-driven sampling and information perturbation to preserve emotion while removing irrelevant factors. To facilitate this process, we propose an emotion clustering and matching approach using emotional attribute prediction and speaker embeddings, enabling generalization to unlabeled data. Additionally, we designed a dual conditioning transformer to integrate style features better. Experimental results confirm the effectiveness of our method in learning speaker-irrelevant emotion embeddings.
![main](https://github.com/user-attachments/assets/2c7e9f52-c14a-47e9-a683-8c1eb35de6c8)


## Training Procedure

### Library
- <a href="https://www.python.org/">Python</a> >= 3.10
- <a href="https://pytorch.org/get-started/pytorch-2.0/">PyTorch</a> >= 2.0 (Recommand)
- <a href="https://pytorch.org/get-started/pytorch-2.0/">CUDA</a> >= 11.6


```bash
  # Docker image
  DOCKER_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
  docker pull $DOCKER_IMAGE

  # Set docker config
  CONTAINER_NAME=YOUR_CONTAINER_NAME
  SRC_CODE=YOUR_CODE_PATH
  TGT_CODE=DOCKER_CODE_PATH
  SRC_DATA=YOUR_DATA_PATH
  TGT_DATA=DOCKER_DATA_PATH
  SRC_CKPT=YOUR_CHECKPOINT_PATH
  TGT_CKPT=DOCKER_CHECKPOINT_PATH
  SRC_PORT=6006
  TGT_PORT=6006
  docker run -itd --ipc host --name $CONTAINER_NAME -v $SRC_CODE:$TGT_CODE -v $SRC_DATA:$TGT_DATA -v $SRC_CKPT:$TGT_CKPT -p $SRC_PORT:$TGT_PORT --gpus all --restart=always $DOCKER_IMAGE
  docker exec -it $CONTAINER_NAME bash

  apt-get update
  # Install tmux
  apt-get install tmux -y
  # Install espeak
  apt-get install espeak -y

  # Clone repository in docker code path
  git clone https://github.com/Choddeok/EmoSpherepp.git

  pip install -r requirements.txt
```

### Vocoder
The BigVGAN 16k checkpoint will be released at a later date. In the meantime, please train using the official BigVGAN implementation or use the official HiFi-GAN checkpoint.
- [[HiFi-GAN]](https://github.com/jik876/hifi-gan)
- [[BigVGAN]](https://github.com/NVIDIA/BigVGAN)

------
### 1. Preprocess data
- Modify the config file to fit your environment.
- We use ESD database, which is an emotional speech database that can be downloaded here: https://hltsingapore.github.io/ESD/.

#### a) Clustering and Matching of Emotion
- The related process can be checked through the following .sh script.
```
sh ./DiEmo_cluster_analyzing/Analyzing.sh
```

#### b) Preprocessing
For binary dataset creation, we follow the pipeline from [[NATSpeech]](https://github.com/NATSpeech/NATSpeech).
```bash
sh preprocessing.sh
```

------
### 2. Training TTS module and Inference  
```bash
sh DiEmoTTS.sh
```

## Acknowledgements
**Our codes are based on the following repos:**
* [NATSpeech](https://github.com/NATSpeech/NATSpeech)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [BigVGAN](https://github.com/NVIDIA/BigVGAN)
