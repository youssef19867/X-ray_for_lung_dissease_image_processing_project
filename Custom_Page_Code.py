from Custom_Filter_Page_Ui import Ui_ImageEnhancementWidget
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget,QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
import numpy as np
import sys
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import os
import traceback
import cv2
import threading
from Custom_KNN_Filter import process_image, visualize_results
      
class custom_filter_widget(QWidget):  # Changed from QMainWindow to QWidget
    def __init__(self):
        super().__init__()
        self.ui = Ui_ImageEnhancementWidget()  # Changed to use Ui_Form instead of Ui_MainWindow
        self.ui.setupUi(self)
        self.dostuff()
    def dostuff(self):
        

        # Example usage
        image_path="images/image_10003_label_Moderate Demented.jpg"
        model_path = "0.90-acc-knn_patch_classifier.pkl"
        original_img = cv2.imread(image_path)
        # Process the image
        processed_img, classification_map, debug_patches, blend_mask = process_image(image_path, model_path)
        visualize_results(original_img, processed_img, classification_map, debug_patches, blend_mask)
        self.original_label = QLabel(self.ui.originalImageFrame)
        self.original_label.setPixmap(QPixmap.fromImage(QImage("images/image_10003_label_Moderate Demented.jpg")))
        self.original_label.setGeometry(0, 0, self.ui.originalImageFrame.width(), self.ui.originalImageFrame.height())
        self.original_label.setScaledContents(True)
        self.area_label = QLabel(self.ui.areaOfImportanceFrame)
        self.area_label.setPixmap(QPixmap.fromImage(QImage("output/classification_map.jpg")))
       
        self.area_label.setGeometry(0, 0, self.ui.areaOfImportanceFrame.width(), self.ui.areaOfImportanceFrame.height())
        self.area_label.setScaledContents(True)  # Optional: if you want the image to fit the label

        # Create a label for the processedImageFrame
        self.processed_label = QLabel(self.ui.enhancedImageFrame)
        self.processed_label.setPixmap(QPixmap.fromImage(QImage("output/processed_image.jpg")))
     
        self.processed_label.setGeometry(0, 0, self.ui.enhancedImageFrame.width(), self.ui.enhancedImageFrame.height())
        self.processed_label.setScaledContents(True) 

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui_ImageEnhancementWidget()
    window.showMaximized()
    sys.exit(app.exec())