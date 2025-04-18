import os
import xml.etree.ElementTree as ET

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

# 例如，指定标注文件的目录路径
ann_dir = r"D:\insect_detection_code\insect_detection_code\anchor\VOCdevkit\VOC2023\Annotations"
class_names = get_class_names(ann_dir)

print(class_names)  # 输出你训练时使用的所有类别，包括背景类
