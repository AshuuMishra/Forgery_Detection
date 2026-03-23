import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import io
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forgery Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }

section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00ff9d;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}

.sub-header {
    font-size: 0.95rem;
    color: #6b6b8a;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.5px;
}

.result-forged {
    background: linear-gradient(135deg, #1a0a0a 0%, #2a0f0f 100%);
    border: 1px solid #ff3366;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.result-authentic {
    background: linear-gradient(135deg, #0a1a0f 0%, #0f2a1a 100%);
    border: 1px solid #00ff9d;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.result-warning {
    background: linear-gradient(135deg, #1a1500 0%, #2a2000 100%);
    border: 1px solid #ffaa00;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.result-forged .result-title   { color: #ff3366; }
.result-authentic .result-title { color: #00ff9d; }
.result-warning .result-title   { color: #ffaa00; }
.result-meta  { font-size: 0.9rem; color: #8888aa; margin-bottom: 0.3rem; }
.result-value { font-family: 'Space Mono', monospace; font-size: 1rem; color: #e8e8f0; font-weight: 700; }

.upload-zone {
    border: 2px dashed #2a2a45;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #6b6b8a;
}

.desc-box {
    background: #12121f;
    border-left: 3px solid #00ff9d;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #aaaacc;
    margin-top: 0.5rem;
}
.desc-box-forged  { border-left-color: #ff3366; }
.desc-box-warning { border-left-color: #ffaa00; }

.type-badge {
    display: inline-block;
    background: #1e1e35;
    border: 1px solid #3a3a5a;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #b0b0d0;
    margin-top: 0.5rem;
}

.warning-box {
    background: #1a1500;
    border: 1px solid #ffaa00;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #ffcc44;
    margin: 0.5rem 0;
}

.info-box {
    background: #0a1020;
    border: 1px solid #0066cc;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #66aaff;
    margin: 0.5rem 0;
}

.supported-box {
    background: #0a150f;
    border: 1px solid #00aa55;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #44cc88;
    margin: 0.3rem 0;
}

.unsupported-box {
    background: #150a0a;
    border: 1px solid #aa3333;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #ff6666;
    margin: 0.3rem 0;
}

.conf-bar-bg            { background: #1e1e35; border-radius: 100px; height: 8px; margin-top: 0.5rem; overflow: hidden; }
.conf-bar-fill-forged   { background: linear-gradient(90deg, #ff3366, #ff6b6b); height: 100%; border-radius: 100px; }
.conf-bar-fill-auth     { background: linear-gradient(90deg, #00ff9d, #00ccff); height: 100%; border-radius: 100px; }
.conf-bar-fill-warning  { background: linear-gradient(90deg, #ffaa00, #ffdd44); height: 100%; border-radius: 100px; }

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

[data-testid="stFileUploader"] {
    background: #12121f;
    border: 1px solid #1e1e35;
    border-radius: 12px;
}

.stButton > button {
    background: #00ff9d;
    color: #0a0a0f;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 0.9rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover { background: #00ccff; transform: translateY(-1px); }
hr { border-color: #1e1e35; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Model Architecture (must match training exactly) ──────────────────────────
class ForgeryDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3, 4]
        )

        with torch.no_grad():
            dummy    = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            channels = features[-1].shape[1]

        self.ela_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels + 64, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.type_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels + 64, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def compute_ela(self, x, quality=90):
        ela_imgs = []
        for img in x:
            try:
                img_np       = (img.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
                pil_img      = Image.fromarray(img_np)
                buf          = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                recompressed = np.array(Image.open(buf)).astype(np.float32)
                ela          = np.abs(img_np.astype(np.float32) - recompressed)
                if ela.max() > 0:
                    ela = ela / ela.max()
                ela_imgs.append(torch.tensor(ela).permute(2, 0, 1).float())
            except Exception:
                ela_imgs.append(torch.zeros(3, 224, 224))
        return torch.stack(ela_imgs).to(x.device)

    def forward(self, x):
        features  = self.backbone(x)
        last_feat = features[-1]
        ela_feat  = self.ela_conv(self.compute_ela(x))
        pooled    = self.global_pool(last_feat).view(x.size(0), -1)
        combined  = torch.cat([pooled, ela_feat], dim=1)
        cls_out   = self.classifier(combined).squeeze(1)
        type_out  = self.type_head(combined)
        seg_out   = self.seg_head(last_feat)
        return cls_out, type_out, seg_out


# ── Constants ─────────────────────────────────────────────────────────────────
TYPE_MAP_INV = {
    0: "none",
    1: "copy_move",
    2: "splicing",
    3: "text_tamper",
    4: "ai_generated"
}

FORGERY_DESCRIPTIONS = {
    "copy_move"   : "A region was copied and pasted within the same document.",
    "splicing"    : "A region from another document was inserted here.",
    "text_tamper" : "A text field was erased and rewritten.",
    "ai_generated": "This image was entirely generated by AI.",
    "none"        : "No forgery detected. Document appears authentic."
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── FIX 1: Image validator — added here, runs before predict ──────────────────
def compute_ela_mean(image_pil):
    """Compute ELA mean to detect unsuitable images (web/screenshot)."""
    img_rgb = np.array(image_pil.convert("RGB"))
    img_rgb = cv2.resize(img_rgb, (224, 224))
    buf     = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = np.array(Image.open(buf)).astype(np.float32)
    ela          = np.abs(img_rgb.astype(np.float32) - recompressed)
    return ela.mean()


def validate_image(image_pil):
    """
    Validates image before prediction.
    Returns (is_valid, warning_message, is_blocked)
    is_blocked=True  → refuse to process
    is_blocked=False → warn but allow
    """
    img_rgb = np.array(image_pil.convert("RGB"))
    h, w    = img_rgb.shape[:2]

    # Check 1 — resolution too low
    if h < 100 or w < 100:
        return False, "Image resolution too low (minimum 100×100 pixels). Please upload a clearer scan.", True

    # Check 2 — aspect ratio (extreme ratios = not a document)
    ratio = h / w
    if ratio < 0.2 or ratio > 6.0:
        return False, "Image dimensions look unusual for a document. Please upload a standard portrait or landscape document.", True

    # Check 3 — ELA mean (web images / screenshots have higher compression noise)
    ela_mean = compute_ela_mean(image_pil)
    if ela_mean > 20:
        return False, (
            f"Image appears to be a screenshot or heavily compressed web image "
            f"(ELA score: {ela_mean:.1f}). This model works best on scanned documents. "
            f"Results would be unreliable."
        ), True

    # Check 4 — low confidence warning zone (ELA 10-20, usable but uncertain)
    if ela_mean > 10:
        return True, (
            f"Image quality is moderate (ELA score: {ela_mean:.1f}). "
            f"For best results, use a direct scan rather than a photo or screenshot."
        ), False

    return True, None, False


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # FIX 2: Load state_dict weights, not full model object
    model_path = "model_weights.pth"
    if not os.path.exists(model_path):
        # Fallback: try old full_model.pth for backward compatibility
        fallback_path = "full_model.pth"
        if os.path.exists(fallback_path):
            try:
                model = torch.load(fallback_path, map_location=DEVICE, weights_only=False)
                model.eval()
                return model
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return None
        return None
    try:
        model = ForgeryDetector().to(DEVICE)
        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ── FIX 3: Predict with threshold=0.40 and confidence gate ───────────────────
def predict(image_pil, model, threshold=0.40):
    image_rgb = np.array(image_pil.convert("RGB"))
    image_rgb = cv2.resize(image_rgb, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tensor = transform(Image.fromarray(image_rgb)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_out, type_out, seg_out = model(tensor)

    confidence = float(torch.sigmoid(cls_out).item())
    confidence = max(0.0, min(1.0, confidence))
    is_forged  = confidence > threshold

    # FIX 4: Confidence gate — flag uncertain predictions
    distance        = abs(confidence - 0.5)
    low_confidence  = distance < 0.15

    # FIX 5: TYPE_MAP_INV fallback to "none" not "unknown"
    forgery_type = TYPE_MAP_INV.get(
        torch.argmax(type_out, dim=1).item(), "none"
    ) if is_forged else "none"

    heatmap         = torch.sigmoid(seg_out).squeeze().cpu().float().numpy()
    if heatmap.ndim == 0:
        heatmap     = np.zeros((224, 224))
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (224, 224))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay         = cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)

    return {
        "is_forged"      : is_forged,
        "confidence"     : confidence,
        "forgery_type"   : forgery_type,
        "description"    : FORGERY_DESCRIPTIONS.get(forgery_type, ""),
        "low_confidence" : low_confidence,
        "image_rgb"      : image_rgb,
        "heatmap"        : heatmap_resized,
        "overlay"        : overlay
    }


# ── Load model on startup ─────────────────────────────────────────────────────
model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 FORGERY DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered document & image tampering analysis</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # FIX 6: Default threshold changed from 0.5 to 0.40
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1, max_value=0.9, value=0.40, step=0.05,
        help="Recommended: 0.40. Higher = stricter forgery detection, may miss some forgeries. Lower = more sensitive, may flag authentic images."
    )

    st.markdown("---")
    st.markdown("### 📊 Forgery Types")
    st.markdown("""
    - 🔄 **Copy-Move** — Region copied within doc
    - ✂️ **Splicing** — Region from another doc
    - ✏️ **Text Tamper** — Erased & rewritten text
    - 🤖 **AI Generated** — Fully AI-generated image
    """)

    st.markdown("---")

    # FIX 7: Added supported/unsupported image guidance in sidebar
    st.markdown("### ✅ Supported Images")
    st.markdown("""
    <div class="supported-box">
    ✓ Scanned prescriptions<br>
    ✓ Scanned medical reports<br>
    ✓ Scanned certificates / IDs<br>
    ✓ Direct scanner output (JPG/PNG)<br>
    ✓ PDF pages exported as image
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ❌ Not Supported")
    st.markdown("""
    <div class="unsupported-box">
    ✗ Screenshots of documents<br>
    ✗ Web downloaded images<br>
    ✗ WhatsApp / Telegram photos<br>
    ✗ Social media images<br>
    ✗ Photos of printed documents
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.success("✅ Model loaded") if model else st.error("❌ Model not found — place model_weights.pth in app folder")
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown("**Model:** EfficientNet B4 + ELA")


col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📁 Upload Image")

    # FIX 8: Added guidance text above uploader
    st.markdown("""
    <div class="info-box">
    ℹ️ For best results upload a <strong>scanned document</strong> saved directly from your scanner.
    Screenshots and web images are not supported.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)
        analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3rem;">📄</div>
            <div style="margin-top:0.5rem;">Drop a scanned document here or click to browse</div>
            <div style="font-size:0.8rem;margin-top:0.3rem;">Supports JPG, PNG, BMP, TIF</div>
            <div style="font-size:0.75rem;margin-top:0.5rem;color:#444466;">
                Prescriptions · Medical Reports · Certificates · ID Documents
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_result:
    st.markdown("### 📋 Analysis Result")

    if uploaded_file and "analyze_btn" in dir() and analyze_btn:
        if not model:
            st.error("⚠️ Model not loaded. Place model_weights.pth in the app folder.")
        else:
            with st.spinner("Validating and analyzing image..."):

                # FIX 9: Validate BEFORE predict
                is_valid, validation_msg, is_blocked = validate_image(image_pil)

                if is_blocked:
                    # Show blocked message — do not run predict
                    st.markdown(f"""
                    <div class="result-warning">
                        <div class="result-title">⚠️ CANNOT PROCESS</div>
                        <div class="desc-box desc-box-warning">
                            {validation_msg}
                        </div>
                        <div style="margin-top:1rem;font-size:0.85rem;color:#aaaacc;">
                            <strong>Please upload:</strong><br>
                            • A scanned prescription or medical document<br>
                            • Saved directly from scanner as JPG or PNG<br>
                            • Not a screenshot or web-downloaded image
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    result = predict(image_pil, model, threshold)

                    is_forged    = result["is_forged"]
                    confidence   = result["confidence"]
                    forgery_type = result["forgery_type"]
                    description  = result["description"]
                    low_conf     = result["low_confidence"]

                    card_class   = "result-forged" if is_forged else "result-authentic"
                    verdict      = "⚠️ FORGED" if is_forged else "✅ AUTHENTIC"
                    bar_class    = "conf-bar-fill-forged" if is_forged else "conf-bar-fill-auth"
                    bar_width    = int(confidence * 100)

                    # FIX 10: Show validation warning if moderate quality
                    if validation_msg:
                        st.markdown(f"""
                        <div class="warning-box">⚠️ {validation_msg}</div>
                        """, unsafe_allow_html=True)

                    # FIX 11: Show low confidence warning
                    if low_conf:
                        st.markdown("""
                        <div class="warning-box">
                        ⚠️ <strong>Low confidence result</strong> — model is uncertain about this image.
                        The result may not be reliable. Consider uploading a higher quality scan.
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="result-title">{verdict}</div>
                        <div class="conf-bar-bg">
                            <div class="{bar_class}" style="width:{bar_width}%"></div>
                        </div>
                        <div style="margin-top:0.8rem;">
                            <div class="result-meta">CONFIDENCE</div>
                            <div class="result-value">{confidence:.1%}</div>
                        </div>
                        <div style="margin-top:0.8rem;">
                            <div class="result-meta">FORGERY TYPE</div>
                            <div class="type-badge">{forgery_type.upper().replace("_"," ")}</div>
                        </div>
                        <div class="desc-box {'desc-box-forged' if is_forged else ''}">
                            {description}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 🗺️ Forgery Heatmap")
                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        fig.patch.set_facecolor("#0a0a0f")
                        ax.set_facecolor("#0a0a0f")
                        ax.imshow(result["heatmap"], cmap="hot")
                        ax.set_title("Suspicion Heatmap", color="#e8e8f0", fontsize=10, pad=8)
                        ax.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    with viz_col2:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        fig.patch.set_facecolor("#0a0a0f")
                        ax.set_facecolor("#0a0a0f")
                        ax.imshow(result["overlay"])
                        ax.set_title("Overlay", color="#e8e8f0", fontsize=10, pad=8)
                        ax.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

    elif not uploaded_file:
        st.markdown("""
        <div style="color:#3a3a5a;text-align:center;padding:3rem 0;
                    font-family:'Space Mono',monospace;font-size:0.9rem;">
            Upload a scanned document and click<br>Analyze to see results here
        </div>
        """, unsafe_allow_html=True)
