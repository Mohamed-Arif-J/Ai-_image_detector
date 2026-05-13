import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# =====================
# 1. Setup
# =====================
MODEL_PATH = "ai_image_detector.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["AI", "Real"]

# =====================
# 2. Image Preprocessing (same as training)
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# 3. Load Model
# =====================
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model

# =====================
# 4. Prediction Function
# =====================
def predict_image(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probs, 0)

    ai_conf = probs[0].item() * 100
    real_conf = probs[1].item() * 100

    print(f"\n🖼 Image: {image_path}")
    print(f"Prediction: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence → AI: {ai_conf:.2f}% | Real: {real_conf:.2f}%")

# =====================
# 5. Run from Command Line
# =====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    predict_image(image_path)
