import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 模型加载函数
def load_model(path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

# 设置路径和类别名
model_path = "D:/insect_detection_code/insect_detection_code/anchor/train.pt"
image_path = r"D:\insect_detection_code\insect_detection_code\anchor\VOCdevkit\VOC2023\JPEGImages\00208_jpg.rf.2bc6fbe9c66f94a4b6e1e0318703b063.jpg"  # 替换为你测试图片实际路径
CLASS_NAMES = ['rice leaf roller', '__background__', 'rice leaf caterpillar']# 加载模型
model = load_model(model_path, len(CLASS_NAMES))

# 加载图片
img = Image.open(image_path).convert("RGB")
transform = transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0)  # 添加 batch 维度

# 推理
with torch.no_grad():
    output = model(img_tensor)[0]

# 可视化结果
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(img)

for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
    if score < 0.5:
        continue
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin, f"{CLASS_NAMES[label]}: {score:.2f}", color='white', backgroundcolor='green')
print(model) 
plt.axis("off")
plt.show()