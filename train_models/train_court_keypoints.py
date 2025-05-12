import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TennisCourtKeyPointsDataset(Dataset):
    def __init__(self, image_folder, data_file):
        self.image_folder = image_folder
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        image = cv2.imread(f"{self.image_folder}/{item['id']}.png")
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        keypoints = np.array(item['kps']).flatten()
        keypoints = keypoints.astype(np.float32)
        keypoints[::2] *= 224 / width
        keypoints[1::2] *= 224 / height
        return image, keypoints


train_dataset = TennisCourtKeyPointsDataset("D:\\tennis_thesis\\data\\images", "/data/data_train.json")
validation_dataset = TennisCourtKeyPointsDataset("D:\\tennis_thesis\\data\\images", "/data/data_val.json")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=True)

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 14*2)
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    for i, (images, keypoints) in enumerate(train_loader):
        images = images.to(device)
        keypoints = keypoints.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "../models/tennis_court_keypoints_model.pth")