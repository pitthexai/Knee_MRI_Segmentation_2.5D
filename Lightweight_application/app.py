import logging
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from pydicom import dcmread, pixel_array

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# .rw-rw-r--   98M amin 25 Aug 16:45   best_model_256_noise_notebook_5_0.15.pth
# .rw-rw-r--   98M amin 25 Aug 16:47   best_model_256_noise_notebook_7_0.10.pth
# .rw-rw-r--   98M amin 25 Aug 16:52   best_model_512_noise_notebook_5_0.15.pth
# .rw-rw-r--   98M amin 25 Aug 16:54   best_model_512_noise_notebook_7_0.10.pt
MODEL_256_PATHS = [
    "models/best_model_256_noise_notebook_5_0.15.pth",
    "models/best_model_256_noise_notebook_7_0.10.pth",
]

MODEL_512_PATHS = [
    "models/best_model_512_noise_notebook_5_0.15.pth",
    "models/best_model_512_noise_notebook_7_0.10.pth",
]

NUM_CLASSES = 5

logger = logging.getLogger("App")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger().setLevel(logging.WARNING)


def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def save_npy_image(image: np.ndarray, path: str):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    img = Image.fromarray(image)

    img.save(path)


def localize(
    image: np.ndarray, model: YOLO
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # type: ignore
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    results = model.predict(image, verbose=False)[0]

    if not results.boxes or len(results.boxes) == 0:
        return None  # No detection

    boxes = results.boxes.xyxy.cpu().numpy().astype(int)  # type: ignore
    scores = results.boxes.conf.cpu().numpy()  # type: ignore

    best_idx = np.argmax(scores)
    x1, y1, x2, y2 = boxes[best_idx]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    best_crop = image[y1:y2, x1:x2]
    # crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]

    return best_crop, (x1, y1, x2, y2)


def preprocess_np_slice(img: np.ndarray, size: int):
    """Convert grayscale np.array to resized torch tensor"""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )
    return transform(pil_img).squeeze(0)  # type: ignore


def stack_three_np_slices(np_slices, index, size, model):
    """Get 3 adjacent slices and stack as 3-channel tensor"""
    slices = [np_slices[i] for i in [index - 1, index, index + 1]]

    localized_slices = []

    for slice in slices:
        res = localize(slice, model)

        if res is None:
            return torch.zeros((1, 3, size, size)).to(DEVICE)

        crop, _ = res
        localized_slices.append(crop)

    slices = [
        preprocess_np_slice(localized_slice, size)
        for localized_slice in localized_slices
    ]

    stacked = torch.stack(slices, dim=0)  # [3, H, W]
    return stacked.unsqueeze(0).to(DEVICE)  # [1,3,H,W]


def run_inference(models, input_tensor, resize_to=None):
    probs = []
    with torch.no_grad():
        for model in models:
            logits = model(input_tensor)
            softmax = F.softmax(logits, dim=1)
            prob = softmax[0]
            if resize_to and (prob.shape[1] != resize_to or prob.shape[2] != resize_to):
                prob = F.interpolate(
                    prob.unsqueeze(0),
                    size=(resize_to, resize_to),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            probs.append(prob.cpu())
    return probs


def fuse_probs(probs_256, probs_512):
    avg_256 = torch.stack(probs_256).mean(dim=0) if probs_256 else None
    avg_512 = torch.stack(probs_512).mean(dim=0) if probs_512 else None

    if avg_256 is not None and avg_512 is not None:
        return (avg_256 + avg_512) / 2
    elif avg_256 is not None:
        return avg_256
    elif avg_512 is not None:
        return avg_512
    else:
        raise ValueError("No model outputs to fuse")


def entropy_map(prob_tensor):
    entropy = -torch.sum(prob_tensor * torch.log(prob_tensor + 1e-12), dim=0)
    return entropy.cpu().numpy()


def overlay_entropy(entropy, image_gray):
    entropy_norm = cv2.normalize(entropy, None, 0, 255, cv2.NORM_MINMAX).astype(  # type: ignore
        np.uint8
    )
    entropy_color = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_JET)

    # Resize grayscale image to match entropy map
    image_gray_resized = cv2.resize(
        image_gray, (entropy_color.shape[1], entropy_color.shape[0])
    )
    image_gray_uint8 = (image_gray_resized * 255).astype(np.uint8)

    image_color = cv2.cvtColor(image_gray_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_color, 0.6, entropy_color, 0.4, 0)
    return overlay


def visualize_segmentation(segmentation):
    colors = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 165, 0],  # ffa500
            [116, 20, 12],  # 74140c
        ],
        dtype=np.uint8,
    )
    return colors[segmentation]


@st.cache_resource(show_spinner=False)
def load_models():
    yolo_model = YOLO("./models/localizer.pt")
    models_256 = [load_model(p) for p in MODEL_256_PATHS]
    models_512 = [load_model(p) for p in MODEL_512_PATHS]
    return yolo_model, models_256, models_512


def load_dicom_series(files):
    """Load and stack a series of DICOM slices into a 3D numpy array"""
    slices = []
    metadata = []

    for f in files:
        ds = dcmread(f)
        arr = ds.pixel_array.astype(np.float32)

        # Normalize to [0,1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        slices.append(arr)
        metadata.append((ds.InstanceNumber if "InstanceNumber" in ds else 0, arr))

    # Sort by InstanceNumber (or fallback to upload order)
    metadata.sort(key=lambda x: x[0])
    slices_sorted = [m[1] for m in metadata]

    volume = np.stack(slices_sorted, axis=0)  # [N, H, W]
    return volume


def main():
    st.set_page_config(layout="wide")

    col_logo, _, col_qr_code = st.columns([1, 5, 1])

    with col_logo:
        st.image("https://pitthexai.github.io/assets/img/Pitthexai_logo.png", width=300)

    with col_qr_code:
        st.image("https://pitthexai.github.io/images/qr-code.png", width=120)

    st.set_page_config(page_title="KneeXNet-2.5D", page_icon="🦿", layout="wide")

    st.title(
        "KneeXNet-2.5D: an AI Tool set for Knee Cartilage and Meniscus Segmentation in MRIs"
    )

    st.markdown("""
    Upload either:
    - A `.npy` file containing MRI slices, OR
    - A folder (multiple files) of `.dcm` DICOM images
    """)

    with st.spinner("Loading models..."):
        yolo_model, models_256, models_512 = load_models()

    uploaded_files = st.file_uploader(
        "Upload NPY or DICOM files", type=["npy", "dcm"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload a file or a folder of DICOM images.")
        return

    try:
        if len(uploaded_files) == 1 and uploaded_files[0].name.endswith(".npy"):
            # Directly load NPY file
            slices_np = np.load(uploaded_files[0])
        else:
            # Assume DICOM series
            dicom_paths = [f for f in uploaded_files if f.name.endswith(".dcm")]
            if not dicom_paths:
                st.error("No valid DICOM files found.")
                return

            slices_np = load_dicom_series(dicom_paths)
            np.save("uploaded_volume.npy", slices_np)  # optional save
            st.success(f"Converted {len(dicom_paths)} DICOM slices into numpy volume")

        if slices_np.ndim != 3 or slices_np.shape[0] < 3:
            st.error("Input must be 3D array: [N_slices, H, W] with at least 3 slices.")
            st.stop()

        num_slices = slices_np.shape[0]
        st.success(f"Loaded volume with {num_slices} slices")

        idx = (
            1
            if num_slices == 3
            else st.slider("Select central slice", 1, num_slices - 2, step=1)
        )

        input_256 = stack_three_np_slices(slices_np, idx, 256, yolo_model)
        input_512 = stack_three_np_slices(slices_np, idx, 512, yolo_model)

        with st.spinner("Running segmentation..."):
            probs_256 = run_inference(models_256, input_256, resize_to=512)
            probs_512 = run_inference(models_512, input_512, resize_to=None)

            fused_prob = fuse_probs(probs_256, probs_512)
            segmentation = torch.argmax(fused_prob, dim=0).cpu().numpy()

        mean_slice = slices_np[idx]
        entropy = entropy_map(fused_prob)
        seg_vis = visualize_segmentation(segmentation)

        col1, col2, col3, col4 = st.columns(4)

        image = mean_slice
        localized_image = np.zeros_like(image)

        res = localize(mean_slice, yolo_model)

        if res is not None:
            _, bbox = res
            x1, y1, x2, y2 = bbox

            localized_image = image[y1:y2, x1:x2]
            localized_image = cv2.resize(localized_image, (512, 512))

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            entropy_overlay = overlay_entropy(entropy, localized_image)
        else:
            entropy_overlay = overlay_entropy(entropy, mean_slice)

        with col1:
            st.image(
                image,
                caption=f"Grayscale Slice {idx}",
                clamp=True,
                use_container_width=True,
            )

        with col2:
            st.image(
                localized_image,
                caption=f"Localized Slice {idx}",
                use_container_width=True,
            )

        with col3:
            st.image(seg_vis, caption="Segmentation Map", use_container_width=True)

        with col4:
            st.image(
                entropy_overlay, caption="Entropy Overlay", use_container_width=True
            )

        cols = st.columns(4)

        with cols[0]:
            st.markdown(
                '<div style="display:flex;align-items:center;">'
                '<div style="background-color:#00FF00;width:20px;height:10px;margin-right:8px;"></div>'
                "<strong>Distal femoral cartilage</strong>"
                "</div>",
                unsafe_allow_html=True,
            )

        with cols[1]:
            st.markdown(
                '<div style="display:flex;align-items:center;">'
                '<div style="background-color:#3399FF;width:20px;height:10px;margin-right:8px;"></div>'
                "<strong>Proximal tibial cartilage</strong>"
                "</div>",
                unsafe_allow_html=True,
            )

        with cols[2]:
            st.markdown(
                '<div style="display:flex;align-items:center;">'
                '<div style="background-color:#FFB300;width:20px;height:10px;margin-right:8px;"></div>'
                "<strong>Patellar cartilage</strong>"
                "</div>",
                unsafe_allow_html=True,
            )

        with cols[3]:
            st.markdown(
                '<div style="display:flex;align-items:center;">'
                '<div style="background-color:#5D1000;width:20px;height:10px;margin-right:8px;"></div>'
                "<strong>Meniscus</strong>"
                "</div>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Failed to process: {e}")


if __name__ == "__main__":
    main()
