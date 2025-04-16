# Bone Fracture Detection Using Deep Learning
## Problem Statement

Bone fractures are a common medical condition.Manual Detection Time-consuming, prone to error.Misdiagnosis can lead to delayed treatments and complications.AI-assisted detection can provide quicker and more accurate results. Assists radiologists in high-workload environments.CNNs (Convolutional Neural Networks) can effectively classify X-ray images as fractured or not.
________________________________________
## Objectives

**Goal:**

Build a binary image classification model to predict if an X-ray image shows a fractured or non-fractured bone.

**Aim:**

• Develop an automated system for detecting bone
fractures from X-ray

• Deployment as a user-friendly web app for real-world
use.

• Improve processing time

• Enhance diagnostic accuracy by reducing human
errors in identifying fractures.
________________________________________
## Dataset

**Source:**

  From Kaggle by Madushani Rodrigo: https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

**Size:**

  Size: 10,580 samples.

  Training: 9,246 images.

  Validation: 828 images.

  Test: 506 images

  Data Type: Image dataset

  Features: image of fractured bone’s x-rays & image of non fractured bone’s x-rays

  **Challenges:** Variability in X-ray image quality
________________________________________
## Methodology

**1. Import required Libraries :**

  o Import necessary libraries for deep learning and image processing.

  o tensorflow, os, keras, matplotlib
  
**2. Data Collection & Preprocessing :**

  o Load Train,Test & validation image dataset.
  
  o Image resizing, normalization (rescale 1./255), augmentation (rotation, flip)
  
**3. Data Visualization:**

  o Visualize x-ray images of both Fractured and Not-fractured.
  
**4. Model Selection:**

  o CNN-based architecture for image classification, with Conv2D, MaxPooling layers, flatten layer, Dense layers.
  
**5. Training :**

  o Fit the model using the training and validation sets.
  
  o 30 epochs with Adam optimizer and binary cross-entropy loss.
  
**6. Evaluation:**
  
  o Accuracy and loss on test data.
  
**7. Deployment:**

  o Streamlit-based web application.
________________________________________  
## Model Implementation

**Algorithm:**

Convolutional Neural Network (CNN)

• 3 Conv2D layers with increasing filters (32→ 64→128)

• MaxPooling to reduce dimensions.

• Flatten, Dense, Dropout layers

• Sigmoid Activation in Output Layer

**Why CNN?**

Effective for image feature extraction.

Suitable for binary classification.

**Training Process:**

• Train-test-validation split via generators

• Hyperparameter tuning (epochs, batch size, learning rate)
________________________________________
## Evaluation & Results

Test Accuracy: 99%

Test Loss: 0.03
________________________________________
## Deployment

**Method:** Web application using Streamlit.

**Technologies:**
Python,TensorFlow/Keras,Streamlit.
________________________________________
## 📂 Project Structure
```
📦 Bone-Fracture-Detection-Using-CNN
├── 📂 app.py         # Contains streamlit application logic and database models
├── 📂 bone.ipynb     # Python file to model development
├── 📜 requirements 
├── 📜 LICENSE           # License file
├── 📜 README.md         # Project Documentation
```
________________________________________
## Challenges & Limitations

Limited Dataset size, Overfitting issues, Long Training time

Model depends on dataset quality
________________________________________
## Key Takeaways:

• AI can aid in medical diagnostics

• CNNs are effective in X-ray image classification
________________________________________
## Future Enhancements:

• Train with larger and more diverse datasets

• Improve model interpretability with heatmaps (Grad-CAM)

• Improve model with advanced architectures (e.g., ResNet, transfer learning)

• Add real-time prediction and integration with medical systems.

• Integration with 3D Imaging: Extending the model to work with CT scans and MRI for more comprehensive fracture
analysis.

• Multi-Disease Diagnosis: Expanding the system to detect other bone-related issues such as osteoporosis or joint
dislocations.
________________________________________
## References

**Dataset Source:**
Dataset obtained from Kaggle by Madushani Rodrigo: https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

**Medium Article:**
Deep learning for X-ray bone fracture detection overview: https://medium.com/@t.mostafid/deep-learning-for-x-ray-bone-fracture-detection-key-concepts-and-recent-approaches-8b6bb509fd8c

**Kaggle Notebook:**
Bone fracture detection using CNN notebook by Mahmoud Ali: https://www.kaggle.com/code/mahmoudalisalem/bonex-bone-fructure-detecion-using-cnn

**Core Libraries:**
TensorFlow and Keras were used for model creation.
The OS library was used for file management.
________________________________________
## Conclusion

This project successfully developed a CNN-based model to detect bone fractures from X-ray
images,

Offering a quick and accurate AI-assisted diagnosis. Deployed using Streamlit.

It provides an accessible tool for medical professionals.

While effective, it faces challenges. This project demonstrates how AI can support faster, more
reliable medical diagnostics, and ultimately enhance patient care
________________________________________
## 🚀 Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammedAjmal786/Bone-Fracture-Detection-Using-CNN.git
   cd Bone-Fracture-Detection-Using-CNN
   ```

2. Install dependencies:
   ```bash
   pip install requirements
   pip install tenserflow
   pip install os
   ```

3. Run the applications:
   - **Streamlit App**:
     ```bash
     cd app.py
     streamlit run app.py
     ```
________________________________________
## Take a peek at what I’ve created! 👀
Streamlit App: https://bone-fracture-detection-using-cnn-537k6mbarrudny3usqfkbj.streamlit.app
________________________________________
## 🤝 Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
________________________________________
## THANK YOU
