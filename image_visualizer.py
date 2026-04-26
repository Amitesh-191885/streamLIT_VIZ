import streamlit as st
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

st.set_page_config(page_title="Image Filter Visualizer", layout="wide")

# =========================================================
# Utility Functions
# =========================================================

def load_image(uploaded_file):
    """Loads an image from an uploaded file and auto-detects if it's Grayscale or RGB."""
    img = Image.open(uploaded_file)
    
    # Resize if too large to prevent freezing
    max_size = 800
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
    # Convert to numpy array
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Auto detect grayscale vs RGB
    if len(img_np.shape) == 2:
        is_rgb = False
    elif len(img_np.shape) == 3:
        if img_np.shape[2] == 4: # Drop alpha channel if present
            img_np = img_np[:, :, :3]
        # Check if R==G==B
        if np.all(img_np[:, :, 0] == img_np[:, :, 1]) and np.all(img_np[:, :, 1] == img_np[:, :, 2]):
            img_np = img_np[:, :, 0]
            is_rgb = False
        else:
            is_rgb = True
    else:
        is_rgb = False
        
    return img_np, is_rgb

def custom_conv2d(img, kernel, padding=0, stride=1):
    """
    Applies 2D convolution with custom padding and stride.
    Handles both 2D (grayscale) and 3D (RGB) images.
    """
    if len(img.shape) == 2:
        # Grayscale
        if padding > 0:
            img_padded = np.pad(img, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        else:
            img_padded = img
            
        filtered = ndimage.convolve(img_padded, kernel, mode='constant', cval=0.0)
        out = filtered[::stride, ::stride]
        return out
        
    elif len(img.shape) == 3:
        # RGB
        out_channels = []
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            if padding > 0:
                img_padded = np.pad(channel, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
            else:
                img_padded = channel
                
            filtered = ndimage.convolve(img_padded, kernel, mode='constant', cval=0.0)
            out = filtered[::stride, ::stride]
            out_channels.append(out)
        return np.stack(out_channels, axis=-1)

def pooling2d(img, pool_size=2, stride=2, pool_type='max'):
    """
    Applies Max or Average pooling.
    """
    # Helper for 2D pooling
    def pool2d_single_channel(ch):
        h, w = ch.shape
        out_h = (h - pool_size) // stride + 1
        out_w = (w - pool_size) // stride + 1
        
        # If output dimensions are <= 0, pooling can't be applied
        if out_h <= 0 or out_w <= 0:
            return ch
            
        out = np.zeros((out_h, out_w))
        for y in range(0, out_h):
            for x in range(0, out_w):
                y_start = y * stride
                x_start = x * stride
                window = ch[y_start:y_start+pool_size, x_start:x_start+pool_size]
                if pool_type == 'max':
                    out[y, x] = np.max(window)
                else:
                    out[y, x] = np.mean(window)
        return out

    if len(img.shape) == 2:
        return pool2d_single_channel(img)
    else:
        out_channels = [pool2d_single_channel(img[:, :, c]) for c in range(img.shape[2])]
        return np.stack(out_channels, axis=-1)

def generate_gaussian_kernel(size=5, sigma=1.0):
    """Generates a 2D Gaussian kernel."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

# =========================================================
# Main Application
# =========================================================

st.title("🖼️ Advanced Image Filter Visualizer")
st.markdown("Upload an image, apply custom convolutions, add pooling, and manipulate stride and padding!")

# Sidebar Controls
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    img, is_rgb = load_image(uploaded_file)
    
    st.sidebar.info(f"Image Auto-detected as: **{'RGB' if is_rgb else 'Grayscale'}**\n\nOriginal Shape: {img.shape}")
    
    st.sidebar.header("2. Processing Task")
    task = st.sidebar.selectbox("Select Filter Task", ["Edge Detection", "Noise Removal", "Band Pass Filter"])
    
    st.sidebar.header("3. Convolution/Pooling Parameters")
    
    # Convolution Params
    st.sidebar.subheader("Convolution")
    padding = st.sidebar.slider("Padding (Zeros)", min_value=0, max_value=5, value=1)
    conv_stride = st.sidebar.slider("Convolution Stride", min_value=1, max_value=5, value=1)
    
    # Pooling Params
    st.sidebar.subheader("Pooling")
    apply_pool = st.sidebar.checkbox("Apply Pooling after Filter?", value=False)
    if apply_pool:
        pool_type = st.sidebar.radio("Pooling Type", ["max", "average"])
        pool_size = st.sidebar.slider("Pooling Window Size", min_value=2, max_value=10, value=2)
        pool_stride = st.sidebar.slider("Pooling Stride", min_value=1, max_value=10, value=2)
    
    # Task Specific Parameters
    st.sidebar.header("Task Configurations")
    
    kernel = None
    processed_img = img.copy()
    
    # -----------------------------
    # 1. Edge Detection
    # -----------------------------
    if task == "Edge Detection":
        edge_type = st.sidebar.radio("Edge Orientation", ["Vertical", "Horizontal"])
        
        # Sobel Kernels
        if edge_type == "Vertical":
            kernel = np.array([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]])
        else:
            kernel = np.array([[-1, -2, -1], 
                               [ 0,  0,  0], 
                               [ 1,  2,  1]])
                               
        processed_img = custom_conv2d(img, kernel, padding=padding, stride=conv_stride)
        
        # Normalize to 0-1 for display
        processed_img = np.abs(processed_img)
        if processed_img.max() > 0:
            processed_img = processed_img / processed_img.max()

    # -----------------------------
    # 2. Noise Removal
    # -----------------------------
    elif task == "Noise Removal":
        noise_type = st.sidebar.radio("Filter Type", ["Gaussian Blur", "Median Filter"])
        
        if noise_type == "Gaussian Blur":
            k_size = st.sidebar.slider("Kernel Size (Gaussian)", 3, 15, 5, step=2)
            sigma = st.sidebar.slider("Sigma (Gaussian)", 0.1, 5.0, 1.0)
            kernel = generate_gaussian_kernel(size=k_size, sigma=sigma)
            processed_img = custom_conv2d(img, kernel, padding=padding, stride=conv_stride)
            processed_img = np.clip(processed_img, 0, 1)
            
        else:
            # Median Filter uses scipy median_filter directly.
            # Stride and Padding are simulated manually.
            m_size = st.sidebar.slider("Window Size (Median)", 3, 15, 3, step=2)
            
            def manual_median(image, size, pad, strd):
                if pad > 0:
                    image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
                med = ndimage.median_filter(image, size=size)
                return med[::strd, ::strd]
                
            if is_rgb:
                out_channels = [manual_median(img[:,:,c], m_size, padding, conv_stride) for c in range(3)]
                processed_img = np.stack(out_channels, axis=-1)
            else:
                processed_img = manual_median(img, m_size, padding, conv_stride)
                
            processed_img = np.clip(processed_img, 0, 1)

    # -----------------------------
    # 3. Band Pass Filter
    # -----------------------------
    elif task == "Band Pass Filter":
        st.sidebar.markdown("*(Using Difference of Gaussians)*")
        k_size = st.sidebar.slider("Kernel Size", 3, 21, 9, step=2)
        sigma1 = st.sidebar.slider("Inner Sigma (High frequency cutoff)", 0.5, 5.0, 1.0)
        sigma2 = st.sidebar.slider("Outer Sigma (Low frequency cutoff)", 1.0, 10.0, 2.0)
        
        kernel1 = generate_gaussian_kernel(size=k_size, sigma=sigma1)
        kernel2 = generate_gaussian_kernel(size=k_size, sigma=sigma2)
        
        # Bandpass is the difference between two gaussians
        kernel = kernel1 - kernel2
        
        processed_img = custom_conv2d(img, kernel, padding=padding, stride=conv_stride)
        
        # Normalize to 0-1 for display
        # Values can be negative, so shift to zero center and scale
        processed_img = processed_img - processed_img.min()
        if processed_img.max() > 0:
            processed_img = processed_img / processed_img.max()

    # -----------------------------
    # Apply Pooling (Optional)
    # -----------------------------
    if apply_pool:
        processed_img = pooling2d(processed_img, pool_size=pool_size, stride=pool_stride, pool_type=pool_type)

    # =========================================================
    # Visualizations
    # =========================================================
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True, clamp=True)
        st.caption(f"Shape: {img.shape}")
        
    with col2:
        st.subheader(f"Processed Image ({task})")
        st.image(processed_img, use_container_width=True, clamp=True)
        st.caption(f"Shape: {processed_img.shape}")
        
    if apply_pool:
        st.info(f"💡 The image was filtered with a Stride of **{conv_stride}** and Padding of **{padding}**, then downsampled using **{pool_type.title()} Pooling** (Window: {pool_size}x{pool_size}, Stride: {pool_stride}).")
    else:
        st.info(f"💡 The image was filtered with a Stride of **{conv_stride}** and Padding of **{padding}**.")
        
else:
    st.info("👈 Please upload an image from the sidebar to get started.")

