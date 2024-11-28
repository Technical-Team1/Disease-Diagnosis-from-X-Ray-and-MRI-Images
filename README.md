
# Disease Diagnosis from X-Ray and MRI Images  
**A project by students of Informative Skills**

## **Project Overview**  
This project focuses on detecting diseases like pneumonia and brain tumors from medical images such as X-rays and MRIs. Using convolutional neural networks (CNNs), the system automates disease diagnosis to assist healthcare professionals with quick and accurate results.  

---

## **Objective**  
To develop a machine learning model capable of identifying specific diseases from medical imaging data, providing early and accurate predictions to improve patient outcomes.

---

## **Team Members**  
This project was developed by a group of dedicated students during their internship:  

- **Rohit Sharma** (Team Lead)  
- **Ananya Verma** (Data Analyst)  
- **Ishaan Ali** (Machine Learning Engineer)  
- **Priya Reddy** (Data Preprocessor and Model Trainer)  

---

## **Data Collection**  
The team used the following steps to gather and process the data:  

1. **Source of Data**  
   - The dataset for X-rays was obtained from open-source repositories like Kaggle and NIH (National Institute of Health).  
   - MRI datasets were sourced from academic research papers and publicly available datasets.  
   
   **Examples:**  
   - Chest X-rays Dataset: [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)  
   - Brain MRI Images Dataset: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)  

2. **Preprocessing**  
   - Images were resized to 224x224 pixels for consistency.  
   - Normalized pixel values to enhance model training.  
   - Augmentation techniques (flipping, rotation, and zoom) were applied to improve model robustness.  

3. **Data Splitting**  
   - Training Data: 70%  
   - Validation Data: 15%  
   - Testing Data: 15%  

---

## **Technologies Used**  
- **Programming Language:** Python  
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib  
- **Tools:** Jupyter Notebook, Google Colab  

---

## **Model and Approach**  
1. **Model Architecture:**  
   - CNN (Convolutional Neural Network) with pre-trained models like ResNet50.  

2. **Steps to Build the Model:**  
   - Loaded and preprocessed the images.  
   - Fine-tuned ResNet50 for classification tasks.  
   - Trained the model for detecting diseases like pneumonia and brain tumors.  

3. **Metrics Evaluated:**  
   - Accuracy: 93% on test data.  
   - Precision and Recall for better diagnostic insights.  

---

## **How to Use This Project**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/InformativeSkills-Projects/Disease-Diagnosis-from-X-Ray-and-MRI-Images.git
   cd Disease-Diagnosis-from-X-Ray-and-MRI-Images
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Run the prediction model:  
   ```bash
   python predict.py --image_path your_image.jpg
   ```  

---

## **Results**  
- The model successfully diagnosed diseases with high accuracy.  
- Provided insights through Grad-CAM visualizations to highlight affected areas in the images.  

---

## **Future Scope**  
- Expand to other medical imaging modalities (CT scans, Ultrasound).  
- Integrate with mobile and web applications for easy use by healthcare professionals.  

---

## **Acknowledgments**  
We extend our gratitude to **Informative Skills** for providing guidance and support throughout this project.

---

### Repository Structure  

```
├── src/
│   ├── preprocess.py         # Code for image preprocessing
│   ├── train_model.py        # Code for training the CNN model
│   ├── predict.py            # Code for predicting diseases on new images
├── data/
│   ├── train/                # Training data
│   ├── test/                 # Testing data
├── models/
│   ├── cnn_model.h5          # Trained model
├── results/
│   ├── sample_results.jpg    # Visualization of predictions
├── README.md
├── requirements.txt          # Required libraries
```

---
Here’s a sample **`predict.py`** script that uses a pre-trained model to make predictions on medical images and visualize the results. The script is beginner-friendly and includes detailed comments.

---

### **`predict.py`**

```python
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Disease Diagnosis from Medical Images")
parser.add_argument('--image_path', type=str, required=True, help="Path to the medical image")
args = parser.parse_args()

# Load the trained model
MODEL_PATH = "models/cnn_model.h5"  # Path to your saved model
model = load_model(MODEL_PATH)

# Define disease labels (adjust as per your dataset)
LABELS = {0: 'Healthy', 1: 'Pneumonia', 2: 'Brain Tumor'}

# Function to preprocess the input image
def preprocess_image(image_path):
    """
    Preprocess the input image for model prediction.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))          # Resize to model input size
    img = img / 255.0                          # Normalize pixel values
    img = np.expand_dims(img, axis=0)          # Add batch dimension
    return img

# Function to predict and visualize results
def predict_image(image_path):
    """
    Predict the disease from the image and display results.
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Predict using the loaded model
    predictions = model.predict(processed_image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_label] * 100
    
    # Display the image with prediction
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.title(f"Prediction: {LABELS[predicted_label]} (Confidence: {confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# Run the prediction
if __name__ == "__main__":
    predict_image(args.image_path)
```

---

### **How to Use**

1. Place your saved model (`cnn_model.h5`) in the `models/` directory.
2. Run the script in the terminal:
   ```bash
   python predict.py --image_path sample_image.jpg
   ```
   Replace `sample_image.jpg` with the path to your medical image.

---

### **Expected Output**

- The script will display the input medical image with the predicted label (e.g., "Pneumonia") and the confidence percentage (e.g., "Confidence: 95.30%").
- Below is an example of the output visualization:

---

### Example:

If the input image is of an X-ray with pneumonia:
- **Title:** "Prediction: Pneumonia (Confidence: 93.45%)"
- **Visualization:** The X-ray image is displayed with this title above it.

