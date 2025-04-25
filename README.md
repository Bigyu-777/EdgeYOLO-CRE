# Lightweight EdgeYOLO-CRE Network for Jaboticaba Tree Recognition Based on Ascend AI Processor
![image](https://github.com/Bigyu-777/EdgeYOLO-CRE/blob/main/EdgeYOLO-CRE.png)
## Introduction
YOLO-CRE is an improved object detection algorithm based on YOLOv5, designed for Jaboticaba tree detection in UAV (Unmanned Aerial Vehicle) imagery. This project replaces YOLOv5's original PAFPN (Path Aggregation Feature Pyramid Network) with CCFM (Cross-Channel Feature Modulation) from RT-DETR and integrates the ECA (Efficient Channel Attention) mechanism to enhance feature extraction and improve detection accuracy.

## Key Improvements
### 1. Replacing PAFPN with CCFM
CCFM improves feature fusion through cross-channel feature modulation. Compared to PAFPN, it enhances information interaction and improves multi-scale object detection performance.

### 2. Introducing ECA Attention Mechanism
ECA utilizes a local cross-channel interaction strategy, avoiding complex fully connected computations while improving channel-wise information flow, enhancing the model’s ability to detect Jaboticaba trees.

### 3. UAV Remote Sensing Data Adaptation
EdgeYOLO-CRE is optimized for the unique visual characteristics of Jaboticaba trees, incorporating tailored data augmentation strategies and anchor box adjustments to enhance performance in UAV imagery.

## Installation
This project requires Python 3.8 or later. It is recommended to create a virtual environment and install dependencies using the following commands:
```bash
# Create a virtual environment
conda create --name EdgeYOLO-CRE python==3.10
conda activate EdgeYOLO-CRE


# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation
1. **Dataset:**
   - The dataset consists of UAV remote sensing images with COCO-style annotations.
   - Managed under the `datasets/` directory.
   - Find [Datase](https://github.com/Bigyu-777/AUG_Jaboticaba_tree)

2. **Directory Structure:**
```shell
EdgeYOLO-CRE/
|-- datasets/
|   |-- images/
|   |   |-- train/
|   |   |-- val/
|   |-- labels/
|   |   |-- train/
|   |   |-- val/
|-- models/
|   |-- cre.yaml  # EdgeYOLO-CRE model configuration
|-- train.py  # Training script
|-- detect.py  # Inference script
|-- README.md  # Project documentation
```

## Training
Train the model using the following command:
```bash
python train.py --img 640 --batch 32 --epochs 30 --data datasets/tree.yaml --cfg models/CRE.yaml 
```
- `--img 640`: Input image size set to 640x640.
- `--batch 32`: Batch size set to 32.
- `--epochs 30`: Train for 30 epochs.
- `--data datasets/tree.yaml`: Dataset configuration file.
- `--cfg models/CRE.yaml`: Custom CRE-YOLO model configuration.
## VAL
Val the model using the following command:
```bash
python val.py --data dataset/tree.yaml --weights ckpt/best.pt --img 640
```

## Inference
Perform object detection using trained weights:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source datasets/images/test/
```

## Results & Performance
Experiments on the UAV dataset demonstrate that EdgeYOLO-CRE achieves higher accuracy than YOLOv5 in Jaboticaba tree detection while maintaining reasonable computational efficiency.

| Model | mAP@0.5 | mAP@0.5:0.95 | Parameters (M) | FPS |
|------|--------|-------------|------------|------------|
| YOLOv5s | 96.4 | 57.8 | 7.2 | 291 |
| EdgeYOLO-CRE | 97.6 | 66.8 | 3.8 | 387 |

---
📄 About Me

Hi, I'm Junyu Huang, a passionate researcher and developer with a strong interest in computer vision, deep learning, and multispectral image processing. I'm currently looking for graduate research opportunities or industry positions related to AI and computer vision.

If you're interested in my background, feel free to check out my [online resume](https://bigyu-777.github.io/resume.html) and [Zh-CN Version](https://bigyu-777.github.io/resume-zh.html).
I'm open to collaborations, research discussions, or any exciting opportunities.

## References
1. RT-DETR: End-to-End Object Detection with Relational Transformers
2. YOLOv5: https://github.com/ultralytics/yolov5
3. ECA: Efficient Channel Attention for Deep Convolutional Neural Networks


