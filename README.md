<img width="1271" height="761" alt="LabelMe answer-card annotation screenshot" src="https://github.com/user-attachments/assets/99ce784a-ebca-49fc-bf9b-1a0785b47ed5" />

# Answer Card Detection Demo

This project is a lightweight computer vision demo for detecting answer-card question numbers and answer options. It converts LabelMe annotations into a YOLO dataset, fine-tunes a YOLO detection model, and exports prediction results back to LabelMe-compatible JSON files for manual review and correction.

## What Problem It Solves

Manual answer-card annotation is repetitive and slow. In this project, a small object detection model is trained to automatically locate:

- `th`: question number area
- `option`: answer option area

After prediction, the project generates one JSON annotation file for each original image. The output can be opened directly in LabelMe, so users only need to correct imperfect boxes instead of drawing every box from scratch.

This workflow is useful for building a data annotation loop:

- Label a small dataset manually
- Fine-tune a detection model
- Use the model to pre-label new images
- Correct the generated labels in LabelMe
- Re-train with better data

## Technologies Used

- Python: command-line training and prediction scripts
- Ultralytics YOLO: object detection training and inference
- PyTorch: deep learning runtime used by YOLO
- OpenCV: image reading, drawing, and output generation
- LabelMe JSON: annotation format for manual correction
- YOLO dataset format: training data format for detection models
- unittest: basic tests for annotation conversion and JSON export

This is a YOLO object detection fine-tuning project, also known as transfer learning. It is not large model pretraining. The model starts from pretrained YOLO weights and is adapted to a custom answer-card detection task.

## Features

- Convert LabelMe rectangle annotations to YOLO labels
- Train two detection classes: `th` and `option`
- Run batch prediction on new images
- Export annotated preview images
- Export LabelMe-style JSON files
- Support a LabelMe-friendly output structure: original image plus same-name JSON

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Training

The training folder should contain image files and same-name LabelMe JSON files:

```text
dataset/
  sample_1.jpg
  sample_1.json
  sample_2.jpg
  sample_2.json
```

Run training:

```powershell
python train.py --source "C:\path\to\labelme_dataset" --epochs 50 --imgsz 640
```

The training script will:

- Convert LabelMe annotations to a YOLO dataset under `datasets/answer_card_yolo`
- Split the data into train and validation sets
- Save training results under `runs/detect/answer_card`

If there is no GPU, CPU training is also supported:

```powershell
python train.py --source "C:\path\to\labelme_dataset" --epochs 1 --device cpu
```

## Prediction

Run batch prediction:

```powershell
python predict.py --weights runs/detect/answer_card/weights/best.pt --input path\to\images --output outputs
```

Output files:

- `outputs/images/`: preview images with detection boxes
- `outputs/json/`: LabelMe-compatible JSON annotations
- `outputs/predictions.csv`: summary of detected boxes

For LabelMe correction, place each original image and its same-name JSON file in the same folder:

```text
labelme_pairs/
  sample_1.jpg
  sample_1.json
  sample_2.jpg
  sample_2.json
```

## Classes

```text
0: th
1: option
```

## Repository Notes

The repository is intended to store source code and project documentation only.

Do not upload:

- Original training images
- Generated datasets
- Prediction outputs
- Training logs
- Model weights

These files are excluded in `.gitignore`:

```text
datasets/
outputs/
runs/
*.pt
```
