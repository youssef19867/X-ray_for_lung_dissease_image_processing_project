from Main_Page_UI import Ui_MainWindow
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import numpy as np
import sys
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from Detail_Page_Code import DetailWidget
from keras.preprocessing import image
from keras.models import load_model
import os
import threading
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.open_file_explorer)
        self.ui.pushButton_5.clicked.connect(self.open_detail_window)
        # Initialize UI elements
        self.ui.label.clear()
        self.ui.label.setParent(self.ui.frame)
        self.ui.label.setGeometry(self.ui.frame.rect())
        self.ui.frame.resizeEvent = self.resize_label
        self.pixmap = None
        self.file_path = None
        # Load model and class name
        
        thread = threading.Thread(target=self.load_model)
        thread.start()
        
        #self.img_size = (150, 150)  # Update based on model's input requirements
    def load_model(self):
        model_path = 'x_ray_model.keras'  # Path to the saved model
        if os.path.exists(model_path):
            model = load_model('chaegaeg.keras', compile=False)
            self.model=model
            print("Model loaded successfully.")
        else:
            print("No saved model found. Please ensure the model is trained and saved.")
            exit()
        self.class_labels = ['covid', 'lung_opacity', 'normal', 'pneumonia']
    def open_detail_window(self):
        self.detail_window = DetailWidget(self.file_path,self)
        self.hide()
        self.detail_window.show()
 
    def resize_label(self, event):
        self.ui.label.setGeometry(self.ui.frame.rect())
        super().resizeEvent(event)
    
    def preprocess_image(self, img_path,target_size=(128, 128)):
        """Process image for model input"""
        try:
             img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')  # Load and resize the image
             
             img_array = image.img_to_array(img)  # Convert to numpy array
             img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
             img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
             
             return img_array
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    def predict_image(self,img_path):
        # Preprocess the image
        img_array = self.preprocess_image(img_path)
        
        # Make a prediction
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
        predicted_class_label = self.class_labels[predicted_class_index]  # Get the class label
        confidence = np.max(predictions)  # Get the confidence score
        
        # Display the image and prediction
        img = image.load_img(img_path)
        
        
        return predicted_class_label, confidence
    def open_file_explorer(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Skin Image", "", 
                "Image Files (*.png *.jpg *.jpeg *.bmp)"
            )

            if not file_path:
                return

            # Validate image file
            if not file_path.lower().endswith(('.png', '.PNG')):
                QMessageBox.warning(self, "Invalid File", "Supported formats: JPG, JPEG")
                return
            self.file_path=file_path
            # Display image
            pixmap = QPixmap(file_path)
            self.pixmap = pixmap
            if pixmap.isNull():
                raise ValueError("Corrupted or unsupported image file")
                
            self.ui.label.setPixmap(pixmap.scaled(
                self.ui.frame.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            self.ui.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Make prediction
            predicted_class, confidence= self.predict_image(file_path)
            msg_box = QMessageBox()  # Create a message box
            msg_box.setWindowTitle("Prediction Result")  # Set the title
            msg_box.setText(f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}")  # Set the message
            
            msg_box.exec()

            # Get proper medical name
    

            # Show diagnostic report
          

        except Exception as e:
            error_msg = (
                f"Error: {str(e)}\n\n"
                "Possible Solutions:\n"
                "1. Use clear, non-blurry images\n"
                "2. Ensure proper lighting\n"
                "3. Capture affected area clearly\n"
                "4. Avoid makeup/creams on skin"
            )
            QMessageBox.critical(self, "Analysis Failed", error_msg)
            print(f"Error details: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.showMaximized()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"Application failed: {str(e)}")
        sys.exit(1)