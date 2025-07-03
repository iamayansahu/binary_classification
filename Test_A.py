import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import os

# ===== Config =====
IMG_SIZE = 224
MODEL_PATH = "Binary_Classification_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['female', 'male']

# ===== Transform for Prediction =====
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===== Model Definition =====
def get_model():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)

# ===== Predict Function =====
def predict_single_image(image_path):
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
        confidence_percentage = prob * 100 if pred == 1 else (1 - prob) * 100

        print(f"🧠 Predicted: {CLASS_NAMES[pred]}")
        print(f"📊 Confidence: {confidence_percentage:.2f}%")

if __name__ == "__main__":
    test_image_path = r"member3.jpg"
    predict_single_image(test_image_path)
