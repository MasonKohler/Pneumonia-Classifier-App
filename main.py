import streamlit as st
import requests
from streamlit_lottie import st_lottie
from keras.models import load_model
from PIL import Image
from util import classify

# -------Streamlit Web Code--------
st.set_page_config(page_title="TKS REP 2", page_icon=":lungs:")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Assets
lottie_xray = load_lottieurl("https://lottie.host/4d3f22af-b69d-4ad7-8726-a7763503b249/pXNnGjgrYk.json")
img_confusion_matrix = Image.open('/Users/masonkohler/PycharmProjects/REP2/images/Confusion Matrix.png')


# Model
with st.container():
    st.title('Pneumonia Image Classifier for Chest X-rays')
    st.header("Please upload X-Ray below: ")
    # Store the uploaded file in a variable
    uploaded_file = st.file_uploader("", type=["jpeg", "jpg", "png"])

    # Load model and process image if a file was uploaded
    if uploaded_file is not None:
        # Load the model
        model = load_model("./model/keras_model.h5")

        # Load labels
        with open('./model/labels.txt', 'r') as f:
            class_names = [a[:-1].split(" ")[1] for a in f.readlines()]

        # Open and display the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)

        # Classify image
        class_name, conf_score = classify(image, model, class_names)

        # Write classification
        st.write("## {}".format(class_name))
        st.write("### confidence: {}%".format(int(conf_score * 1000) / 10))

# About the app
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("About this app")
        st.write("##")
        st.write(
            """
            This model was trained using Google's Teachable Machine and uploaded to a StreamLit web-app using Python and TensorFlow. 

            The goal of this project is to showcase how far AI in medical image has come, and how it can impact the future of radiology. 
            
            
            """
        )
    with right_column:
        st_lottie(lottie_xray, height=300, key="xray")

#Statistics about the model


with st.container():
    st.write("---")
    st.header("Model Statistics :bar_chart:")
    st.write("##")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "91.5%")
    with col2:
        st.metric("Sensitivity", "93.3%")
    with col3:
        st.metric("Specificity", "87.2%")
    with col4:
        st.metric("Total Cases", "676")

with st.container():
    st.write("---")
    st.header("Confusion Matrix")
    st.write("###")
    st.write("""

            Below is a Confusion Matrix, showing the performance of the model. Let me break down what the numbers mean:

            True Positives (448): The model correctly identified 448 cases of pneumonia

            False Positives (25): The model incorrectly classified 25 normal X-rays as pneumonia

            False Negatives (32): The model missed 32 cases of pneumonia, classifying them as normal

            True Negatives (171): The model correctly identified 171 normal X-rays


            This data was used to calculate the metrics seen in the columns above:
           
                Accuracy: (448 + 171)/(448 + 25 + 32 + 171) ≈ 91.5%
                Sensitivity (True Positive Rate): 448/(448 + 32) ≈ 93.3%
                Specificity (True Negative Rate): 171/(171 + 25) ≈ 87.2%

            The model appears to be performing well, but has a slight bias towards diagnosing pneumonia (more false positives than false negatives). This might be acceptable in a medical context where missing a case of pneumonia (false negative) is generally considered worse than incorrectly flagging a normal case for further review (false positive).

            """)

    st.write("##")
    st.image(img_confusion_matrix)


# Resources
with st.container():
    st.write("---")
    st.header("Resources :books:")
    st.write("##")
    st.write(
        "This model was trained on the Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification dataset by Daniel Kermany, Kang Zhang and Michael Goldbaum.")
    st.write("[Find it here](https://data.mendeley.com/datasets/rscbjbr9sj/2)")
    st.write(
        "Try training your own image classification models using Google's Teachable Machine " + "[Here](https://teachablemachine.withgoogle.com)")

# About me
with st.container():
    st.write("---")
    st.header("About me")
    st.write("##")
    st.write(
        """
        My name is Mason Kohler, and I'm a grade 11 student passionate about the future of AI in medical imaging. If you would like to reach me for any other projects or questions, don't hesitate to send me an email at: mason.wm.kohler@gmail.com. Thank you for checking this project out, I hope you learned something new. 
        """
    )