import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# --- Setup ---
st.set_page_config(page_title="AI Image Detector")
st.title(" AI Image Detector")
st.write("Upload an image to detect whether it’s **AI-generated or Real Artwork** — with confidence score and chart.")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model_path = "ai_image_detector.pth"

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 2)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # --- Prediction ---
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    classes = ["AI Generated", "Real Artwork"]
    confidence, predicted = torch.max(probabilities, 0)
    label = classes[predicted.item()]
    percent = confidence.item() * 100

    # --- Results ---
    st.markdown(f"### ✅ Prediction: **{label}**")
    st.progress(int(percent))
    st.markdown(f"**Confidence:** {percent:.2f}%")

    # --- Breakdown ---
    st.write("### 🔍 Probability Breakdown:")
    st.write(f"🧠 AI Generated: {probabilities[0].item() * 100:.2f}%")
    st.write(f"🎨 Real Artwork: {probabilities[1].item() * 100:.2f}%")

    # --- Chart ---
    fig, ax = plt.subplots()
    ax.bar(classes, [probabilities[0].item() * 100, probabilities[1].item() * 100],
           color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Model Prediction Confidence")
    ax.set_ylim(0, 100)
    for i, v in enumerate([probabilities[0].item() * 100, probabilities[1].item() * 100]):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10, weight='bold')
    st.pyplot(fig)
