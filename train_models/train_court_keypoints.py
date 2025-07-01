import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json
import numpy as np
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TennisCourtKeyPointsDataset(Dataset):
    def __init__(self, image_folder, data_file):
        self.image_folder = image_folder
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
        keypoints = np.array(item['kps']).flatten().astype(np.float32)
        keypoints[::2] *= 224 / width
        keypoints[1::2] *= 224 / height
        return image, keypoints


train_dataset = TennisCourtKeyPointsDataset("D:\\tennis_thesis\\data\\images", "D:\\tennis_thesis\\data\\data_train"
                                                                               ".json")
validation_dataset = TennisCourtKeyPointsDataset("D:\\tennis_thesis\\data\\images", "D:\\tennis_thesis\\data"
                                                                                    "\\data_val.json")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, keypoints in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]"):
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, keypoints in val_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs("../models", exist_ok=True)
        torch.save(model.state_dict(), "../models/tennis_court_keypoints_model.pth")
        print(f"Model saved at epoch {epoch + 1} with val_loss {avg_val_loss:.4f}")
