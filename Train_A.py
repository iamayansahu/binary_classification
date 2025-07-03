import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# ===== Configuration =====
DATA_DIR = r"Task_A_Dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
MODEL_PATH = "Binary_Classification_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['female', 'male']

# ===== Transforms =====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===== Load Dataset =====
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

# ===== Compute Sample Weights for Sampler =====
labels = [label for _, label in train_data]
class_counts = np.bincount(labels)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ===== Data Loaders =====
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# ===== Class Weights for Loss =====
computed_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
computed_class_weights = torch.tensor(computed_class_weights, dtype=torch.float).to(DEVICE)

# ===== Model Definition =====
def get_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)

model = get_model()
criterion = nn.BCEWithLogitsLoss(pos_weight=computed_class_weights[1])
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ===== Training Function =====
def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"📦 Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# ===== Evaluation Function =====
def evaluate():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.astype(int).flatten())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

if __name__ == "__main__":
    train()
    evaluate()