<img width="1271" height="761" alt="image" src="https://github.com/user-attachments/assets/99ce784a-ebca-49fc-bf9b-1a0785b47ed5" />

# 图片答题卡识别 Demo

这是一个最小可运行的答题卡目标检测项目，适合把现有的 LabelMe 标注数据转换成 YOLO 数据集进行训练，并对新图片批量生成可继续在 LabelMe 中微调的 JSON 标注。

当前项目只做目标检测，不做 OCR、不判断正确答案，也不建立题号与选项之间的业务关系。

## 功能

- LabelMe JSON 转 YOLO 检测数据集
- 训练 `th` 和 `option` 两个类别
- 批量推理新图片
- 导出 LabelMe 风格 JSON
- 支持整理成 `原图 + 同名 JSON` 的 LabelMe 配对目录

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 训练

训练集要求是：

- 图片文件，例如 `xxx.jpg`
- 同名 LabelMe 标注，例如 `xxx.json`

运行训练：

```powershell
python train.py --source "C:\Users\22234\Desktop\kgt\训练集" --epochs 50 --imgsz 640
```

训练会自动：

- 生成本地 YOLO 数据集到 `datasets/answer_card_yolo`
- 默认按 `8:2` 划分训练集和验证集
- 输出训练结果到 `runs/detect/answer_card`

如果没有 GPU，可以显式指定 CPU：

```powershell
python train.py --source "C:\Users\22234\Desktop\kgt\训练集" --epochs 1 --device cpu
```

## 推理

对新图片批量生成标注：

```powershell
python predict.py --weights runs/detect/answer_card/weights/best.pt --input path\to\images --output outputs
```

输出内容：

- `outputs/images/`：带框预览图
- `outputs/json/`：同名 LabelMe JSON
- `outputs/predictions.csv`：检测结果汇总

如果你后续要直接用 LabelMe 微调，推荐把原始图片和导出的 JSON 放在同一个目录中，形成：

- `xxx.jpg`
- `xxx.json`

## 类别

```text
0: th
1: option
```


