from matplotlib.pyplot import text
import streamlit as st
import json
import numpy as np
from tensorflow.python.ops.gen_array_ops import empty
from demo import *
from PIL import Image
import math


@st.cache(allow_output_mutation=True)
def init():
    demo = LayoutNetDemo(checkpoint_path='./checkpoints/ckpt-300')
    return demo


def main():
    st.beta_set_page_config(page_title='LayoutNet', page_icon=None, layout='centered', initial_sidebar_state='auto')

    demo = init()
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('LayoutNet in TensorFlow2.3')

    # 
    image_1_placeholder = st.empty()
    image_2_placeholder = st.empty()

    # read category and corresponding keywords from json
    f = open('./dataset/keywords.json')
    category_keywords_dict = json.load(f)

    category = st.sidebar.selectbox(label='Category', options=tuple(category_keywords_dict.keys()))
    txt_ratio = st.sidebar.slider(label='Text Ratio', min_value=0.1, max_value=0.7, step=0.1, value=0.5)
    img_ratio = st.sidebar.slider(label='Image Ratio', min_value=0.1, max_value=1., step=0.1, value=0.5)

    upload_image_1 = st.sidebar.file_uploader("Choose image 1", type=['png', 'jpg'], encoding='auto')

    images_group = []

    if upload_image_1 is not None:
        image_1_placeholder.image(upload_image_1, width=600)
        images_group.append(Image.open(upload_image_1))
    
    upload_image_2 = st.sidebar.file_uploader("(Optional) Choose image 2", type=['png', 'jpg'], encoding='auto')
    
    if upload_image_2 is not None:
        image_2_placeholder.image(upload_image_2, width=600)
        images_group.append(Image.open(upload_image_2))

    keywords_str = st.sidebar.text_input(label='Keywords (split by ,)')

    # handle keywords string
    keywords = keywords_str.replace(' ', '')
    keywords = keywords.split(',')

    number_of_results = st.sidebar.slider(label='Number of Results', min_value=1, max_value=16, step=1, value=5)
    
    generate = st.sidebar.button(label='Generate')

    if generate:
        canva_row = round(math.sqrt(number_of_results))
        canva_col = math.ceil(float(number_of_results) / canva_row)

        canva = np.zeros((64 * canva_row, 64* canva_col, 3), dtype=np.uint8)

        for i in range(number_of_results):
            row_idx = int(i / canva_col)
            col_idx = int(i % canva_col)
            z = np.random.normal(0.0, 1.0, size=(1, config.z_dim)).astype(np.float32)
            image_raw = demo.generate(category, txt_ratio, img_ratio, images_group, keywords, z)

            canva[row_idx * 64 : row_idx * 64 + 64, col_idx * 64 : col_idx * 64 + 64, :] = np.uint8(image_raw * 255)

        image = Image.fromarray(canva)

        st.image(image, width=600)

if __name__ =='__main__':
    main()
