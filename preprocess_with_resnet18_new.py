import os
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.io import savemat

# 数据预处理和特征提取
def load_data(domain_path, domain_name, max_samples=4000):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # 修改关键点
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    features = []
    labels = []
    classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']


    for cls_idx, cls in enumerate(classes):
        cls_path = os.path.join(domain_path, domain_name, cls)
        samples = 0
        for img_name in os.listdir(cls_path):
            if samples >= max_samples // 7:
                break
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(img).cpu().numpy().flatten()
                features.append(feat)
                labels.append(cls_idx)
                samples += 1
            except:
                continue

    features = np.array(features)
    labels = np.array(labels)
    print(features.shape, labels.shape)
    # 保存为.mat文件
    mat_filename = os.path.join(domain_path, f"{domain_name}.mat")
    savemat(mat_filename, {
        'feas': features,  # (n_samples, n_features)
        'label': labels.reshape(-1, 1)  # (n_samples, 1)
    })

# 主程序
if __name__ == '__main__':
    data_path = './PACS'  # 假设数据集路径
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']

    # 调用load_data，将所有数据转换为.mat文件
    for domain in domains:
        load_data(data_path, domain)


