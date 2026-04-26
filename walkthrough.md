# Streamlit Image Filter Visualizer

The application is complete and is currently running! You can view it by going to `http://localhost:8501` in your browser. 

## What Was Accomplished

I created the new file `image_visualizer.py` implementing all the required tasks. I opted for **pure NumPy and SciPy** as you requested, so no extra dependencies like OpenCV needed to be installed.

### 1. Image Auto-Detection
The application uses Pillow to upload the image and mathematically checks its channels. It auto-detects if the image is Grayscale or RGB and processes it accordingly without requiring a manual toggle.

### 2. Custom Convolution and Pooling
Since you requested the ability to change **Stride** and **Padding** dynamically (which standard libraries don't natively expose well together), I wrote custom implementations for `conv2d` and `pooling2d` that let you manipulate those parameters in real-time using Streamlit sliders.

### 3. Image Processing Tasks
- **Edge Detection**: You can apply Vertical and Horizontal Sobel filters.
- **Noise Removal**: 
  - **Gaussian Blur**: Implemented via a custom-generated Gaussian kernel.
  - **Median Filter**: Leverages SciPy's median filter, adapted to support custom stride and padding.
- **Band Pass Filters**: Implemented using the **Difference of Gaussians** technique. You can tune the Inner Sigma and Outer Sigma to control which frequency bands are preserved.

## How to Test
1. Make sure Streamlit is running (I have started it for you in the background).
2. Open your browser and navigate to the Streamlit local URL.
3. Upload an image from the sidebar.
4. Try switching the "Filter Task" between Edge Detection, Noise Removal, and Band Pass.
5. Play with the Padding, Stride, and Pooling sliders and notice how the Output Image Shape gets modified as it reduces the spatial resolution of the image.

> [!TIP]
> If you upload very large images, the app automatically scales them down to a maximum width of 800px to ensure the custom convolution loops run quickly and smoothly.
