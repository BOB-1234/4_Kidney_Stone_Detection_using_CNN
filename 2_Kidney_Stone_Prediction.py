#importing the necessary classes to detect and upload the trained machine learning model
import numpy as np
from PIL import Image
from keras.models import load_model

#identifying the image size for the testing for detection
IMAGE_SIZE = 150

# Can upload either kidney stone or healthy images by uncommenting and commenting the provided list of images for testing the uploaded machine learnig model

# Load the image and resize it to the desired size patient with kidney stone (5 different cases provided below for testing patient with kidney stone)
#image_name = 'Dataset/Test/Normal/Normal- (1004).jpg'
#image_name = 'Dataset/Test/Normal/Normal- (1005).jpg'
#image_name = 'Dataset/Test/Normal/Normal- (1006).jpg'
#image_name = 'Dataset/Test/Normal/Normal- (1007).jpg'
image_name = 'Dataset/Test/Normal/Normal- (1008).jpg'

# Can upload healthy patient data by uncommneting the provided images below
#image_name = 'Dataset/Test/Stone/Stone- (1178).jpg'
#image_name = 'Dataset/Test/Stone/Stone- (1177).jpg'
#image_name = 'Dataset/Test/Stone/Stone- (1176).jpg'
#image_name = 'Dataset/Test/Stone/Stone- (1175).jpg'
#image_name = 'Dataset/Test/Stone/Stone- (1174).jpg'

# Resize the image uploaded it to the desired size patient with kidney stone
img = Image.open(image_name).resize((IMAGE_SIZE, IMAGE_SIZE))

# Calling the model created to named kidney_stone_model.h5 to predict the uploaded image from before
model = load_model('kidney_stone_model.h5')

# Function to Predict CNN based on feature identification
def predict_btn_cnn(image):
    test_img = np.expand_dims(image, axis=0)
    result = model.predict(test_img)
    if result[0][0] == 1:
        print("Patient has kidney stone")
    elif result[0][0] == 0:
        print("Patient is Healthy")

# Call the created function with the final result printed
predict_btn_cnn(img)
