from Custom_Filter_Page_Ui import Ui_ImageEnhancementWidget
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
import numpy as np
import sys
import joblib
from PIL import Image
import os
import traceback
import cv2
import threading
import matplotlib
# Force matplotlib to use a non-interactive backend that works with threads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Custom_KNN_Filter import process_image, visualize_results

class custom_filter_widget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ImageEnhancementWidget()
        self.ui.setupUi(self)
        
        # Initialize progress bar to 0
        self.ui.progressBar.setValue(0)
        
        # Start the processing with a small delay to allow UI to initialize
        QTimer.singleShot(100, self.start_processing)
    
    def start_processing(self):
        # Create a timer that updates every second (1000ms)
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        
        # Set up variables for progress tracking
        self.progress_value = 0
        self.max_progress = 100
        self.processing_time = 1  # Total time in seconds
        self.progress_increment = self.max_progress / self.processing_time
        
        # Start the timer
        self.progress_timer.start(1000)  # Update every 1 second
        
        # Start the processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_image_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def update_progress(self):
        self.progress_value += self.progress_increment
        self.ui.progressBar.setValue(int(self.progress_value))
        
        # Stop the timer if we reach max progress
        if self.progress_value >= self.max_progress:
            self.progress_timer.stop()
    
    def process_image_thread(self):
        try:
            # Example usage
            image_path = "images/image_10003_label_Moderate Demented.jpg"
            model_path = "0.90-acc-knn_patch_classifier.pkl"
            
            # Process the image
            original_img = cv2.imread(image_path)
            processed_img, classification_map, debug_patches, blend_mask = process_image(image_path, model_path)
            
            # Save visualizations to files (instead of showing them directly)
            visualize_results(original_img, processed_img, classification_map, debug_patches, blend_mask)
            
            # Once processing is complete, update the UI in the main thread
            QTimer.singleShot(0, self.update_ui)
        except Exception as e:
            print(f"Error in processing thread: {e}")
            traceback.print_exc()
    
    def update_ui(self):
        try:
            # Create and set original image label
            self.original_label = QLabel(self.ui.originalImageFrame)
            self.original_label.setPixmap(QPixmap.fromImage(QImage("images/image_10003_label_Moderate Demented.jpg")))
            self.original_label.setGeometry(0, 0, self.ui.originalImageFrame.width(), self.ui.originalImageFrame.height())
            self.original_label.setScaledContents(True)
            self.original_label.show()
            
            # Create and set area of importance label
            self.area_label = QLabel(self.ui.areaOfImportanceFrame)
            self.area_label.setPixmap(QPixmap.fromImage(QImage("output/classification_map.jpg")))
            self.area_label.setGeometry(0, 0, self.ui.areaOfImportanceFrame.width(), self.ui.areaOfImportanceFrame.height())
            self.area_label.setScaledContents(True)
            self.area_label.show()
            
            # Create and set processed image label
            self.processed_label = QLabel(self.ui.enhancedImageFrame)
            self.processed_label.setPixmap(QPixmap.fromImage(QImage("output/processed_image.jpg")))
            self.processed_label.setGeometry(0, 0, self.ui.enhancedImageFrame.width(), self.ui.enhancedImageFrame.height())
            self.processed_label.setScaledContents(True)
            self.processed_label.show()
            
            # Set progress to 100% when complete
            self.ui.progressBar.setValue(100)
        except Exception as e:
            print(f"Error updating UI: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = custom_filter_widget()
    window.show()
    window.showMaximized()
    sys.exit(app.exec())