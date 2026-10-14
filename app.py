import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AnimalVision AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DARK PROFESSIONAL THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #080b12;
    color: #e8edf7;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background: #0d111a;
    border-right: 1px solid #202838;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */

.hero-title {
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 5px;
    color: #f5f7fb;
}

.hero-title span {
    color: #6ea8fe;
}

.hero-subtitle {
    font-size: 18px;
    line-height: 1.7;
    color: #9ca8ba;
    max-width: 850px;
    margin-bottom: 30px;
}

/* Badge */

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    background: #111a2a;
    border: 1px solid #263a5c;
    color: #7fb0ff;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Cards */

.card {
    background: #101620;
    border: 1px solid #202a3a;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 21px;
    font-weight: 700;
    color: #f1f5fb;
    margin-bottom: 10px;
}

.card-text {
    color: #9ca8ba;
    line-height: 1.7;
    font-size: 15px;
}

/* Metrics */

.metric-card {
    background: #101620;
    border: 1px solid #202a3a;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.metric-number {
    font-size: 30px;
    font-weight: 800;
    color: #78aaff;
}

.metric-label {
    color: #8996aa;
    font-size: 13px;
}

/* Upload */

[data-testid="stFileUploader"] {
    background: #101620;
    border: 1px dashed #34445d;
    border-radius: 16px;
    padding: 15px;
}

/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #314a70;
    background: #15243b;
    color: #eaf1ff;
    font-weight: 600;
}

/* Prediction */

.prediction-box {
    background: #111b2b;
    border: 1px solid #31517f;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-top: 20px;
}

.prediction-name {
    font-size: 35px;
    font-weight: 800;
    color: #7fb0ff;
}

.confidence {
    font-size: 17px;
    color: #b7c4d8;
    margin-top: 8px;
}

/* Footer */

.footer {
    text-align: center;
    color: #667286;
    font-size: 13px;
    padding-top: 40px;
    padding-bottom: 20px;
}

hr {
    border-color: #202a3a;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "Bear",
    "Bird",
    "Cat",
    "Cow",
    "Deer",
    "Dog",
    "Dolphin",
    "Elephant",
    "Giraffe",
    "Horse",
    "Kangaroo",
    "Lion",
    "Panda",
    "Tiger",
    "Zebra"
]


# ============================================================
# VGG7 MODEL
# ============================================================

class VGG7(nn.Module):

    def __init__(self, num_classes=15):

        super(VGG7, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(
            3, 32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(32)

        # Block 2
        self.conv3 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            64, 64,
            kernel_size=3,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(64)

        # Block 3
        self.conv5 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            padding=1
        )

        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(
            128, 128,
            kernel_size=3,
            padding=1
        )

        self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(
            128 * 4 * 4,
            num_classes
        )

    def forward(self, x):

        x = torch.relu(
            self.bn1(self.conv1(x))
        )

        x = torch.relu(
            self.bn2(self.conv2(x))
        )

        x = self.pool(x)

        x = torch.relu(
            self.bn3(self.conv3(x))
        )

        x = torch.relu(
            self.bn4(self.conv4(x))
        )

        x = self.pool(x)

        x = torch.relu(
            self.bn5(self.conv5(x))
        )

        x = torch.relu(
            self.bn6(self.conv6(x))
        )

        x = self.pool(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

device = torch.device("cpu")

model = VGG7(
    num_classes=len(CLASSES)
)

model_loaded = False
model_error = None

try:

    checkpoint = torch.load(
        "vgg7_best.pth",
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    model_loaded = True

except Exception as e:

    model_error = str(e)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🐾 AnimalVision")

    st.markdown("---")

    st.markdown("### Project")

    st.write("Animal Image Classification")

    st.markdown("### Model")

    st.write("VGG7 Convolutional Neural Network")

    st.markdown("### Classes")

    st.write("15 Animal Categories")

    st.markdown("### Input")

    st.write("224 × 224 RGB Image")

    st.markdown("### Framework")

    st.write("PyTorch + Streamlit")

    st.markdown("---")

    st.caption("AI Image Classification System")


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="badge">DEEP LEARNING • COMPUTER VISION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Animal<span>Vision</span> AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
    An intelligent image classification system powered by a custom VGG7
    convolutional neural network. Upload an animal image and let the model
    identify the most probable category with a confidence score.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">15</div>
        <div class="metric-label">Animal Classes</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">VGG7</div>
        <div class="metric-label">CNN Architecture</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">224²</div>
        <div class="metric-label">Input Resolution</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">CPU</div>
        <div class="metric-label">Inference Device</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ============================================================
# PROJECT INFORMATION
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Architecture",
    "Requirements",
    "Future Scope"
])


# ============================================================
# OVERVIEW TAB
# ============================================================

with tab1:

    st.markdown("## Project Overview")

    st.markdown("""
    <div class="card">
        <div class="card-title">Problem Statement</div>
        <div class="card-text">
        Manual identification of animals from images can be time-consuming
        when dealing with large collections of visual data. This project
        develops an automated computer vision system that classifies an
        uploaded animal image into one of 15 predefined categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Objectives</div>
        <div class="card-text">
        • Build a CNN-based image classification system.<br>
        • Train a custom VGG7 architecture for animal recognition.<br>
        • Preprocess images into a consistent 224 × 224 format.<br>
        • Predict the animal category from an unseen image.<br>
        • Display prediction confidence through an interactive interface.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Supported Animal Categories")

    cols = st.columns(5)

    for i, animal in enumerate(CLASSES):

        with cols[i % 5]:

            st.markdown(
                f"""
                <div class="card" style="padding:14px;text-align:center;">
                    {animal}
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ARCHITECTURE TAB
# ============================================================

with tab2:

    st.markdown("## VGG7 Architecture")

    st.markdown("""
    <div class="card">
        <div class="card-title">Convolutional Pipeline</div>
        <div class="card-text">
        Input Image → Convolution → Batch Normalization → ReLU →
        Pooling → Convolution → Batch Normalization → ReLU →
        Pooling → Convolution → Batch Normalization → ReLU →
        Pooling → Adaptive Average Pooling → Fully Connected Layer
        </div>
    </div>
    """, unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("""
        <div class="card">
        <div class="card-title">Block 1</div>
        <div class="card-text">
        Conv2D: 3 → 32<br>
        Conv2D: 32 → 32<br>
        Batch Normalization<br>
        Max Pooling
        </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="card">
        <div class="card-title">Block 2</div>
        <div class="card-text">
        Conv2D: 32 → 64<br>
        Conv2D: 64 → 64<br>
        Batch Normalization<br>
        Max Pooling
        </div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="card">
        <div class="card-title">Block 3</div>
        <div class="card-text">
        Conv2D: 64 → 128<br>
        Conv2D: 128 → 128<br>
        Batch Normalization<br>
        Max Pooling
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Model Configuration")

    st.write("""
    - Input: RGB image
    - Input size: 224 × 224
    - Convolution kernel: 3 × 3
    - Pooling: 2 × 2 Max Pooling
    - Adaptive Average Pooling: 4 × 4
    - Dropout: 0.5
    - Output classes: 15
    """)


# ============================================================
# REQUIREMENTS TAB
# ============================================================

with tab3:

    st.markdown("## Project Requirements")

    r1, r2 = st.columns(2)

    with r1:

        st.markdown("""
        <div class="card">
        <div class="card-title">Software</div>
        <div class="card-text">
        • Python 3.12<br>
        • PyTorch<br>
        • Torchvision<br>
        • Streamlit<br>
        • Pillow<br>
        • NumPy<br>
        • VS Code<br>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:

        st.markdown("""
        <div class="card">
        <div class="card-title">Machine Learning Pipeline</div>
        <div class="card-text">
        • Image preprocessing<br>
        • Dataset preparation<br>
        • CNN training<br>
        • Validation<br>
        • Model evaluation<br>
        • Model saving<br>
        • Image inference<br>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Image Preprocessing")

    st.code("""
Resize → 224 × 224
       ↓
Convert to Tensor
       ↓
ImageNet Normalization
       ↓
VGG7 Model
       ↓
15-Class Prediction
    """)


# ============================================================
# FUTURE SCOPE
# ============================================================

with tab4:

    st.markdown("## Future Scope")

    st.markdown("""
    <div class="card">
        <div class="card-title">Possible Improvements</div>
        <div class="card-text">
        • Increase dataset size and diversity.<br>
        • Apply stronger data augmentation.<br>
        • Experiment with transfer learning.<br>
        • Improve validation and test performance.<br>
        • Deploy the model on cloud infrastructure.<br>
        • Add real-time camera classification.<br>
        • Support additional animal categories.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CLASSIFIER
# ============================================================

st.markdown("---")

st.markdown("## Try AnimalVision AI")

st.markdown(
    """
    <div class="hero-subtitle">
    Upload an animal image below. The trained VGG7 model will process the
    image and return the predicted animal category together with its
    confidence score.
    </div>
    """,
    unsafe_allow_html=True
)


if not model_loaded:

    st.error("Model could not be loaded.")

    st.code(model_error)

else:

    uploaded_file = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:

            st.image(
                image,
                caption="Uploaded Image",
                width=450
            )

        with right:

            image_tensor = transform(image)

            image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)

            with torch.no_grad():

                outputs = model(image_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = CLASSES[
                predicted.item()
            ]

            confidence_value = (
                confidence.item() * 100
            )

            st.markdown(
            f"""<div class="prediction-box">
            <div style="color:#9ca8ba;">PREDICTED ANIMAL</div>
            <div class="prediction-name">{predicted_class}</div>
            <div class="confidence">Confidence: {confidence_value:.2f}%</div>
            </div>""",
            unsafe_allow_html=True
            )

        # Top 3

        st.markdown("### Top 3 Predictions")

        top_values, top_indices = torch.topk(
            probabilities[0],
            3
        )

        for value, index in zip(
            top_values,
            top_indices
        ):

            animal = CLASSES[index.item()]

            percentage = value.item() * 100

            st.write(
                f"**{animal}** — {percentage:.2f}%"
            )

            st.progress(
                float(value.item())
            )


# =========================
# HOW IT WORKS
# =========================

st.markdown("---")

st.header("How It Works")

st.write("1. Upload an animal image")
st.write("2. The image is resized to 224 × 224")
st.write("3. The VGG7 CNN extracts visual features")
st.write("4. The model predicts one of 15 animal classes")
st.write("5. The predicted animal and confidence are displayed")


# =========================
# PROJECT REQUIREMENTS
# =========================

st.header("Project Requirements")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Software")
    st.write("""
    • Python 3.12  
    • PyTorch  
    • Torchvision  
    • Streamlit  
    • Pillow  
    • NumPy  
    • VS Code
    """)

with col2:
    st.subheader("Model")
    st.write("""
    • VGG7 CNN  
    • 15 animal classes  
    • 224 × 224 input image  
    • CPU inference  
    • ImageNet normalization
    """)


# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "AnimalVision AI • CNN-Based Animal Image Classification"
)