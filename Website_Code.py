import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=[128,128])
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Disease Information","FAQs and Help"])


if (app_mode == 'Home'):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_image.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_column_width=True)

    # Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)

        # Reading Labels
        class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
            'Apple___healthy', 'Blueberry___healthy', 
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
            'Corn_(maize)___healthy', 'Grape___Black_rot', 
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
            'Peach___Bacterial_spot', 'Peach___healthy', 
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
            'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
            'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        predicted_disease_name = class_names[result_index]
        st.success("Model is Predicting it's a {}".format(predicted_disease_name))
        st.header('Plant Disease Information')

        # Load the CSV file with disease details
        file_path = 'detailed_plant_disease_details.csv'  # Update this path if needed
        
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            st.error("CSV file not found. Please check the file path.")
            st.stop()  # Stop execution if file not found
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()  # Stop execution on other errors

        # Normalize the column names
        data.columns = data.columns.str.strip()  # Remove leading and trailing spaces

        # Filter data based on the predicted disease name
        disease_data = data[data['Disease Name'] == predicted_disease_name]

        # Display the predicted disease and its details
        if not disease_data.empty:
            st.write("**Causes:**")
            st.write(disease_data['Causes'].values[0])  # Display the cause of the predicted disease
            st.write("**Fertilizer or Pesticide to be Used:**")
            st.write(disease_data['Fertilizer or Pesticide'].values[0])  # Display the recommended fertilizer or pesticide
        else:
            st.write(f"No data available for the predicted disease: {predicted_disease_name}")

elif app_mode == "Disease Information":
    st.header("Disease Information")
    
    # Load the CSV file with disease details
    file_path = 'detailed_plant_disease_details.csv'  # Update this path if needed

    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("CSV file not found. Please check the file path.")
        st.stop()  # Stop execution if file not found
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()  # Stop execution on other errors

    # Normalize the column names
    data.columns = data.columns.str.strip()  # Remove leading and trailing spaces

    # Create a select box for disease names
    disease_names = data['Disease Name'].unique()
    selected_disease = st.selectbox("Select a Disease", disease_names)

    # Filter data based on the selected disease name
    disease_info = data[data['Disease Name'] == selected_disease]

    # Display the selected disease and its details
    if not disease_info.empty:
        st.write("**Selected Disease:** {}".format(selected_disease))
        st.write("**Causes:**")
        st.write(disease_info['Causes'].values[0])  # Display the cause of the selected disease
        st.write("**Fertilizer or Pesticide to be Used:**")
        st.write(disease_info['Fertilizer or Pesticide'].values[0])  # Display the recommended fertilizer or pesticide
    else:
        st.write(f"No data available for the selected disease: {selected_disease}")

elif app_mode == "FAQs and Help":
    st.header("FAQs and Help")
    st.markdown("""
    ### Frequently Asked Questions (FAQs)

    **1. What is the Plant Disease Recognition System?**  
    The Plant Disease Recognition System is a web application designed to help users identify plant diseases by analyzing images of plant leaves. The system uses advanced machine learning algorithms to provide accurate predictions and recommendations for treatment.

    **2. How do I use the app?**  
    To use the app, navigate to the **Disease Recognition** page, upload an image of a plant leaf with suspected disease, and click the **Predict** button. The system will analyze the image and display the predicted disease along with causes and recommended treatments.

    **3. What types of plants and diseases can the system recognize?**  
    The system is trained to recognize a variety of plants and their associated diseases. Currently, it supports the following plants and diseases: 
    - Apple: Apple scab, Black rot, Cedar apple rust
    - Tomato: Late blight, Early blight, Bacterial spot
    - And many more! For a full list, please refer to the disease information section.

    **4. What should I do if the app does not recognize a disease?**  
    If the app cannot identify a disease or provides an incorrect prediction, please double-check the image quality. Ensure that the plant leaf is clearly visible, well-lit, and free from obstructions. You can also provide feedback to help improve the model.

    **5. How can I find information about a specific disease?**  
    You can navigate to the **Disease Information** section and use the dropdown menu to select a specific disease name. The app will then display information about the disease's causes and recommended fertilizers or pesticides.

    **6. What are the recommended fertilizers and pesticides?**  
    The app provides specific recommendations for fertilizers and pesticides based on the predicted disease. These recommendations are designed to help manage the identified diseases effectively.

    **7. Is there a limit to the number of images I can upload?**  
    Currently, there is no strict limit on the number of images you can upload for analysis. However, please allow some time for processing if you upload multiple images at once.

    **8. Who can I contact for further support?**  
    If you have any additional questions or need further assistance, feel free to contact our support team at [helpplantdisease@gmail.com] or use the contact form provided in the app.

    ### Help Section

    **Tips for Uploading Images**
    - **Image Quality**: Ensure the images are clear and in focus. Avoid blurry or low-resolution photos for better prediction accuracy.
    - **Lighting**: Take photos in well-lit conditions to capture the true colors and details of the leaves.
    - **Background**: Use a plain background to minimize distractions and improve recognition accuracy.

    **Troubleshooting Common Issues**
    - **Image Not Uploading**: If your image fails to upload, check your internet connection and try refreshing the page.
    - **Slow Processing Time**: Depending on your internet speed and the server load, processing time may vary. Please be patient while the model analyzes your image.
    - **Unexpected Predictions**: If you receive an unexpected result, consider the quality of the image and the angles from which the photo was taken. Retrain the model with new data for improved accuracy if necessary.

    **Feedback and Improvement**
    We welcome your feedback to improve the app! If you encounter issues or have suggestions for enhancements, please let us know through the contact form or email.

    Thank you for using the Plant Disease Recognition System! 🌱
    """)
