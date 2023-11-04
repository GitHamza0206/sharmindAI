import cv2
import sys
import json
import torch
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import gc 
from transformers import pipeline
import zipfile
import os
import base64
from rembg import remove

from io import BytesIO


warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ShardMind AI",
    page_icon="🚀",
    layout= "wide",
    )

logo = Image.open('./logo.png')
st.sidebar.image(logo)

# Function to zip files
def zip_files(filenames):
    zip_filename = "cropped_images.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in filenames:
            zipf.write(file, arcname=os.path.basename(file))
    return zip_filename

# Function to get a download link for the zip file
def get_zip_download_link(zip_filename):
    with open(zip_filename, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download="{zip_filename}">Download all cropped images</a>'
        st.markdown(href, unsafe_allow_html=True)


def crop_to_transparent_images(pil_image, masks):
    cropped_images = []
    for mask in masks:
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))

        bbox = mask_image.getbbox()
        if bbox:
            cropped_image = pil_image.crop(bbox)
            cropped_image_with_transparency = Image.new("RGBA", cropped_image.size)
            mask_in_bbox = mask_image.crop(bbox)
            cropped_image_with_transparency.paste(cropped_image, mask=mask_in_bbox)
            cropped_images.append(cropped_image_with_transparency)

    return cropped_images

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    byte_data = buffered.getvalue()
    st.download_button(label=text, data=byte_data, file_name=filename, mime='image/png')

def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks):
    fig, ax = plt.subplots()
    ax.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    st.pyplot(fig)
    del mask
    gc.collect()
     
@st.cache_data()
def process_image(image):
    
    pil_image = Image.fromarray(image)

    # Run the segmentation
    mask_generator = pipeline("mask-generation", model="facebook/sam-vit-base")
    outputs = mask_generator(pil_image)
    masks = outputs['masks']

    # Crop the images based on the masks
    cropped_images = crop_to_transparent_images(pil_image, masks)

    return pil_image, masks, cropped_images

st.title("✨ ShardMind AI 🏜")
st.info(' Let me help generate segments for any of your images. 😉')


image_path = st.sidebar.file_uploader("Upload Image 🚀", type=["png","jpg","bmp","jpeg"])
if image_path is not None:
    with st.spinner("Working.. 💫"):

        image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image, masks, cropped_images = process_image(image)

        # Display original and masked images
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            show_masks_on_image(pil_image, masks)  # Assuming this function overlays masks on the image

                # Display a gallery of cropped images
        st.write("Cropped Images Gallery:")
        # Set the number of columns for the gallery
        num_columns = 3
        cols = st.columns(num_columns)
        for i, cropped_image in enumerate(cropped_images):
            # Calculate the column index
            col_index = i % num_columns
            # Display the image in the appropriate column
            with cols[col_index]:
                st.image(cropped_image, use_column_width=False)

        # Save cropped images to a temporary directory and collect file paths
        saved_file_paths = []
        for i, cropped_image in enumerate(cropped_images):
            filename = f"cropped_image_{i}.png"
            cropped_image.save(filename)
            saved_file_paths.append(filename)


        # # Provide download links for cropped images
        # st.write("Download Cropped Images:")
        # for i, cropped_image in enumerate(cropped_images):
        #     get_image_download_link(
        #         cropped_image, 
        #         f"cropped_image_{i}.png", 
        #         f"Download cropped image {i}"      
        #     )


        zip_filename = zip_files(saved_file_paths)
        get_zip_download_link(zip_filename)

        # Clean up individual files after zipping
        for file in saved_file_paths:
            os.remove(file)
else:
    st.warning('⚠ Please upload your Image! 😯')

