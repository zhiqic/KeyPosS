# KeyPosS: Facial Landmark Detection through GPS-Inspired True-Range Multilateration 

KeyPosS is a novel facial landmark detection method inspired by GPS technology. It addresses limitations of traditional heatmap and coordinate regression techniques with an efficient and accurate approach.

<div align="center">
  <img src='https://github.com/zhiqic/KeyPosS/assets/65300431/33f4aab7-d809-443e-a150-56e518246b2c' width='900'/>
  <br>
  <i>Figure 1: A comparison of four decoding methods. Our KeyPosS excels with minimal overhead.</i>
</div>

KeyPosS uses a fully convolutional network to predict distance maps between points of interest (POIs) on a face and multiple anchor points. The anchor points are then leveraged to precisely triangulate the POIs' positions using true-range multilateration.

<div align="center">
  <img src='https://github.com/zhiqic/KeyPosS/assets/65300431/145ed88a-428c-446b-955c-c386c40a489c' width='900'/>
  <br>
  <i>Figure 2: The KeyPosS pipeline, encompassing the Distance Encoding Model, Station Anchor Sampling Strategy, and True-range Multilateration. A versatile scheme suitable for any distance encoding-based approach.</i>
</div>

## Key Features

- **GPS-inspired:** Applies proven concepts from GPS technology to facial analysis, enabling more precise localization.

- **True-Range Multilateration:** Decodes predicted distances into landmark coordinates through multilateration with anchoring points.

- **Versatile:** Can be built upon any distance encoding-based model for enhanced performance.

- **Efficient:** Avoids computational burdens of heatmap-based methods.

For more details, please see our [ACM MM 2023 paper](https://arxiv.org/abs/2305.16437).


## Performance Overview

<div align="center">
  <img src='https://github.com/zhiqic/KeyPosS/assets/65300431/b70c97fd-8ddd-4ec5-8220-8691f2cb2544' width='900'/>
  <br>
  <i>Table 1: A performance comparison with State-of-the-Art methods. Results are presented in NME (%), with top results in bold.</i>
</div>




## Quick Start Guide

Get started with the KeyPosS facial landmark detection system in a few simple steps:

### 1. Installation:

- **Environment Setup**: Begin by setting up the necessary environment. For this, refer to the instructions provided by [mmpose](https://github.com/open-mmlab/mmpose).
  
- **Datasets**: Our experiments utilize the COCO, WFLW, 300W, COFW, and AFLW datasets.

### 2. Training:

- **Pre-trained Models**: We leverage ImageNet models from [mmpose](https://github.com/open-mmlab/mmpose) as our starting point.

- **Training Command**: To start the training process, execute the following command:

  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_train.sh \
      configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256_dark.py \
      4 \
      --work-dir exp/exp889
  ```

### 3. Evaluation:

#### Step 1: Obtain the Models
- **Download**: Retrieve the pre-trained and trained models for each dataset and heatmap resolution from [Google Drive](https://drive.google.com/drive/folders/1gIH6GCVdSH7K0O1_Ohy9sJCRNdsQjnwY).

#### Step 2: Model Setup
- **Placement**: After downloading, move the "exp" model file to the root directory of your codebase.

#### Step 3: Resolution Configuration
- **Supported Resolutions**: The model in the "exp" directory is compatible with five resolutions: 64, 32, 16, 8, and 4.
  
- **Configuration**: Prior to running the test script, adjust the resolution by editing the "data_cfg/heatmap_size" field in the configuration file to your chosen resolution.

#### Step 4: Test Execution
- **Script Selection**: Based on your chosen resolution, run the appropriate test script:
  - `run_test_64.sh`
  - `run_test_32.sh`
  - `run_test_16.sh`
  - `run_test_8.sh`
  - `run_test_4.sh`
  
  These scripts evaluate the model's efficacy across various face datasets: WFLW, COCO, 300W, AFLW, and COFW.

#### Step 5: Evaluation Command
- **Command Execution**: To kick off the evaluation, input the following command:

  ```shell
  CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
      configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256_dark.py \
      exp/exp_v1.3.0/best_NME_epoch_60.pth \
      4 
  ```

## Acknowledgment
Our work is primarily based on [mmpose](https://github.com/open-mmlab/mmpose). We express our gratitude to the authors for their invaluable contributions.

## Citation
If you find this work beneficial, kindly cite our paper:
```bibtex
@inproceedings{bao2023keyposs,
  title={KeyPosS: Plug-and-Play Facial Landmark Detection through GPS-Inspired True-Range Multilateration},
  author={Bao, Xu and Cheng, Zhi-Qi and He, Jun-Yan and Xiang, Wangmeng and Li, Chenyang and Sun, Jingdong and Liu, Hanbing and Liu, Wei and Luo, Bin and Geng, Yifeng and others},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={5746--5755},
  year={2023}
}
```

## License
This repository is licensed under the Apache 2.0 license. For more details, please refer to the LICENSE file.
