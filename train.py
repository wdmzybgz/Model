import os
import torch
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 路径配置
DATASET_DIR = r"D:\insect_detection_code\insect_detection_code\anchor\VOCdevkit\VOC2023"
IMG_DIR = os.path.join(DATASET_DIR, "JPEGImages")
ANN_DIR = os.path.join(DATASET_DIR, "Annotations")

# 动态获取类别名称
def get_class_names(ann_dir):
    class_names = set(["__background__"])  # 包含背景类
    for ann_file in os.listdir(ann_dir):
        if ann_file.endswith(".xml"):
            ann_path = os.path.join(ann_dir, ann_file)
            tree = ET.parse(ann_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text
                class_names.add(name)
    return list(class_names)

# 获取数据集类别
CLASS_NAMES = get_class_names(ANN_DIR)
print("Class Names:", CLASS_NAMES)

# 定义数据集类
class VOCDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        # 返回数据集中的图像数量
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace(".jpg", ".xml"))

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            label = CLASS_NAMES.index(name)
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if len(boxes) == 0:
            # 如果没有目标框，返回空的字典
            target = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64)}
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

# 数据增强函数
def get_transform():
    return T.Compose([T.ToTensor()])

# 加载模型
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 定义 collate_fn 函数
def collate_fn(batch):
    return tuple(zip(*batch))

# 训练主函数
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用 GPU

    dataset = VOCDataset(IMG_DIR, ANN_DIR, transforms=get_transform())
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    model = get_model(num_classes=len(CLASS_NAMES))
    model.to(device)  # 将模型移动到 GPU 或 CPU

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    scaler = GradScaler()  # 用于混合精度训练

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = list(img.to(device) for img in images)  # 图像数据在设备上
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # 目标数据在设备上

            optimizer.zero_grad()

            # 使用混合精度训练
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()  # 缩放损失，反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新 scaler

            total_loss += losses.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f}")
        lr_scheduler.step()

    torch.save(model.state_dict(), r"D:\insect_detection_code\insect_detection_code\anchor\train.pt")
    print("✅ 训练完成，模型已保存")

if __name__ == "__main__":
    train()
