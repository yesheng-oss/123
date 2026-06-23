# 图片答题卡自动标注与检测 Demo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-green)
![LabelMe](https://img.shields.io/badge/LabelMe-JSON-orange)


<img width="1271" height="761" alt="LabelMe answer-card annotation screenshot" src="https://github.com/user-attachments/assets/99ce784a-ebca-49fc-bf9b-1a0785b47ed5" />

## 项目解决的问题

在答题卡、试卷截图、题目切片等场景中，如果完全依赖人工使用 LabelMe 逐个绘制题号和选项框，效率会比较低，而且重复劳动很多。本项目通过训练一个轻量级目标检测模型，自动定位：

- `th`：题号区域
- `option`：选项区域

模型推理后会为每张原始图片生成一个同名 JSON 文件，目录形式类似：

```text
labelme_pairs/
  sample_1.jpg
  sample_1.json
  sample_2.jpg
  sample_2.json
```

这样可以直接用 LabelMe 打开图片和 JSON，人工只需要修正不准确的框，而不是从零开始标注。

## 使用的技术

- `Python`：实现训练、推理和数据转换脚本
- `Ultralytics YOLO`：用于目标检测模型训练和预测
- `PyTorch`：YOLO 模型运行和训练的深度学习框架
- `OpenCV`：读取图片、绘制检测框、保存预览图
- `LabelMe JSON`：作为人工标注和后期微调的标注格式
- `YOLO Dataset Format`：作为模型训练所需的数据格式
- `unittest`：用于测试标注转换和 JSON 导出逻辑

更准确地说，这是一个“基于 YOLO 预训练权重的目标检测微调 / 迁移学习项目”，不是大模型预训练。模型并不是从零开始训练，而是在已有 YOLO 权重基础上适配答题卡题号和选项检测任务。

## 核心流程

```text
LabelMe 标注数据
        ↓
转换为 YOLO 数据集
        ↓
微调 YOLO 检测模型
        ↓
批量预测新图片
        ↓
导出 LabelMe JSON
        ↓
人工微调修正标注
```

## 功能特性

- 支持 LabelMe 矩形框 JSON 转 YOLO 标签
- 支持训练 `th` 和 `option` 两类目标
- 支持批量识别新图片
- 支持输出带框预览图
- 支持输出 LabelMe 风格 JSON
- 支持整理为 `原图 + 同名 JSON` 的 LabelMe 可编辑目录

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 训练模型

训练数据目录需要包含图片和同名 LabelMe JSON：

```text
dataset/
  sample_1.jpg
  sample_1.json
  sample_2.jpg
  sample_2.json
```

运行训练：

```powershell
python train.py --source "C:\path\to\labelme_dataset" --epochs 50 --imgsz 640
```

训练脚本会自动：

- 将 LabelMe JSON 转换为 YOLO 数据集
- 默认按 `8:2` 划分训练集和验证集
- 将训练结果保存到 `runs/detect/answer_card`

如果没有 GPU，也可以使用 CPU：

```powershell
python train.py --source "C:\path\to\labelme_dataset" --epochs 1 --device cpu
```

## 批量预测与自动标注

```powershell
python predict.py --weights runs/detect/answer_card/weights/best.pt --input path\to\images --output outputs
```

输出内容：

- `outputs/images/`：带检测框的预览图
- `outputs/json/`：LabelMe 可编辑 JSON
- `outputs/predictions.csv`：检测结果汇总

## 类别定义

```text
0: th
1: option
```

## 仓库说明

本仓库只保存项目源码和说明文档，不上传训练数据、推理结果和模型权重。

以下内容已在 `.gitignore` 中排除：

```text
datasets/
outputs/
runs/
*.pt
```
