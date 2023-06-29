import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
from load_model import loadModel, imshow
import numpy as np
from PIL import Image
from torchvision import transforms

#wide layout
st. set_page_config(layout="wide")

#remove the menu sign
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


#horizontal menu
selected = option_menu(
    menu_title="Image Enhancer",
    options=["HOME", "ABOUT", "VIDEO"],
    icons=["house", "book", "envelope", "gear"],
    menu_icon = None,
    default_index=0,
    orientation="horizontal",
)

if selected == "HOME":
    #format
    left, right = st.columns(2)
    with left:
      image_comparison(
        img1="https://cdn.britannica.com/25/172925-050-DC7E2298/black-cat-back.jpg",
        img2="https://upload.wikimedia.org/wikipedia/commons/b/bc/Juvenile_Ragdoll.jpg",
        label1="Before",
        label2="After",
        width=600,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

    with right:
      #title
      st.title('AI IMAGE ENHANCER/DEBLUR TOOL')
      st.text("Just upload your blurry image to enhance it without losing quality!")
      uploaded_file = st.file_uploader("Choose a file")
      if uploaded_file is not None:
        out = loadModel(uploaded_file)
        if out is not None:
            print("output is set")
        # To read file as bytes:
        # bytes_data = out.getvalue()
        # st.write(bytes_data)
        #
        # # To convert to a string based IO:
        # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)
        #
        # # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)
        #
        # # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)
        outPer = out.permute(1, 2, 0)
        detached_tensor = outPer.detach()
        outArray = detached_tensor.numpy()
        outArrayNorm = (outArray - np.min(outArray)) / (np.max(outArray) - np.min(outArray))
        #outArrayNorm.shape
        st.image(outArrayNorm, caption=None, width=500)
        # im = unloader(out)
        # buf = BytesIO()
        # im.save(buf, format="PNG")
        # byte_im = buf.getvalue()
        # st.download_button(label="Download Sharp Image", data="sharpImage.png", file_name="sharpImage.png", mime=None)

    st.title('')
    st.markdown("<h1 style='text-align: center; color: white;'>ENHANCE YOUR IMAGE</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
      # render image-comparison
      image_comparison(
        img1="https://static.scientificamerican.com/sciam/cache/file/32665E6F-8D90-4567-9769D59E11DB7F26_source.jpg",
        img2="https://upload.wikimedia.org/wikipedia/commons/b/bc/Juvenile_Ragdoll.jpg",
        label1="Before",
        label2="After",
        width=400,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

    with col2:
      # render image-comparison
      image_comparison(
        img1="https://static.scientificamerican.com/sciam/cache/file/32665E6F-8D90-4567-9769D59E11DB7F26_source.jpg",
        img2="https://upload.wikimedia.org/wikipedia/commons/b/bc/Juvenile_Ragdoll.jpg",
        label1="Before",
        label2="After",
        width=400,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

    with col3:
      # render image-comparison
      image_comparison(
        img1="https://static.scientificamerican.com/sciam/cache/file/32665E6F-8D90-4567-9769D59E11DB7F26_source.jpg",
        img2="https://upload.wikimedia.org/wikipedia/commons/b/bc/Juvenile_Ragdoll.jpg",
        label1="Before",
        label2="After",
        width=400,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
        in_memory=True,
    )

if selected == "ABOUT":
    st.title("")

    left, right = st.columns(2)
    with left:
        st.markdown("<h3 style='text-align: center; color: white;'>Mohak Acharya</h1>", unsafe_allow_html=True)
    with right:
        st.text("""I am Mohak, CV instructor at AI Camp, I have been a mentor to all these
students while developing this Deblur project, I have helped them at
several points and guided them to correct sources when they needed my
help.""")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.text("""Hi, my name's Alfredo and I am from the Bay Area. I mostly worked
on the backend for this project, as well as the connection between
the front end and back. """)
    with right:
        st.markdown("<h3 style='text-align: center; color: white;'>Alfredo Fernandez</h1>", unsafe_allow_html=True)

    st.divider()

    left, right = st.columns(2)
    with left:
        st.markdown("<h3 style='text-align: center; color: white;'>Micheal Li</h1>", unsafe_allow_html=True)
    with right:
        st.text("""Hello! My name is Micheal and I live in Chine. I helped write the
code for the model and train it. I mostly worked on the backend. """)

    st.divider()

    left, right = st.columns(2)
    with left:
        st.text("""Hi! My name's Freya and I live in Gilbert, Arizona. I contributed
to this project by creating the frontend website you are looking
at right now! I also helped collect images for training the model.""")
    with right:
        st.markdown("<h3 style='text-align: center; color: white;'>Freya Bajaj</h1>", unsafe_allow_html=True)

    st.divider()

    left, right = st.columns(2)
    with right:
        st.text("""Hello! My name's Calixte de Belloy and I am 14 years old. I Live in
Marin which is just north of San Francisco. I contributed to the project
by collecting more than 500 images to train the model and give data to the
backhand coders.""")
    with left:
        st.markdown("<h3 style='text-align: center; color: white;'>Calixte de Belloy</h1>", unsafe_allow_html=True)

    st.divider()

    left, right = st.columns(2)
    with right:
        st.markdown("<h3 style='text-align: center; color: white;'>Aditya Shinde</h1>", unsafe_allow_html=True)
    with left:
        st.text("""Hey, my name's Aditya and I am from Folson, California. In this
project, I helped collect images and contributed to the backend as
well as connecting the backend to the frontend.""")

if selected == "VIDEO":
    st.title("You have selected Video")