from Main_Page_UI import Ui_MainWindow
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QListView,QLabel
from PyQt6.QtGui import QPixmap, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread,QStringListModel

import numpy as np
import sys
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from Detail_Page_Code import DetailWidget
from keras.preprocessing import image
from keras.models import load_model
import os
import requests
from bs4 import BeautifulSoup
import threading
import requests
from bs4 import BeautifulSoup
from PyQt6.QtCore import QObject, pyqtSignal
import requests
from bs4 import BeautifulSoup
from PyQt6.QtCore import QObject, pyqtSignal
import re

class ScraperWorker(QObject):
    finished = pyqtSignal(list, str)

    def __init__(self, predicted_class, scrape_type):
        super().__init__()
        self.predicted_class = predicted_class.lower()
        self.scrape_type = scrape_type

    def run(self):
        try:
            data = self.scrape_wikipedia(self.scrape_type)
            
             
        except Exception as e:
            print(f"Error in {self.scrape_type} scraper: {str(e)}")
            data = [f"Error fetching {self.scrape_type} data."]
        self.finished.emit(data, self.scrape_type)
        

    def format_condition(self):
        return self.predicted_class.replace('_', ' ').replace(' ', '_')

    def scrape_wikipedia(self,thing):
        try:
            condition = self.format_condition()
            print(f"Scraping data for condition: {condition}")
            url = f"https://en.wikipedia.org/wiki/{condition}"

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return [f"Failed to retrieve Wikipedia page for {condition} (Status: {response.status_code})"]

            soup = BeautifulSoup(response.text, 'html.parser')

            paragraphs = soup.find_all('p')

            matched_paragraphs = [
                p.get_text(strip=True) for p in paragraphs
                if thing in p.get_text().lower() and condition.lower() in p.get_text().lower()
            ]

            return matched_paragraphs if matched_paragraphs else [f"No relevant 'causes'or 'treatment' information found for {condition}."]
        
        except requests.exceptions.RequestException as e:
            return [f"Network error: {str(e)}"]
        except Exception as e:
            return [f"Scraping error: {str(e)}"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.open_file_explorer)
        self.ui.pushButton_5.clicked.connect(self.open_detail_window)
        self.ui.label.clear()
        self.ui.label.setParent(self.ui.frame)
        self.ui.label.setGeometry(self.ui.frame.rect())
        self.ui.pushButton_2.clicked.connect(self.save_cause_of_disease)
        self.ui.pushButton_3.clicked.connect(self.save_treatment_of_disease)
        self.ui.frame.resizeEvent = self.resize_label
        self.pixmap = None
        self.file_path = None
        
        thread = threading.Thread(target=self.load_model)
        thread.start()
        
    def save_cause_of_disease(self):
        try:
           
            model = self.ui.listView_2.model()
            
          
            if model is None or model.rowCount() == 0:
                QMessageBox.warning(self, "Warning", "No causes data to save")
                return
            
       
            items = []
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                items.append(model.data(index))
            
      
            with open("cause_of_disease.txt", "w", encoding="utf-8") as f:
                for item in items:
                    f.write(item + "\n\n")  
            
            QMessageBox.information(self, "Success", "Causes data saved to 'cause_of_disease.txt'")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save causes: {str(e)}")
    def save_treatment_of_disease(self):
        try:
            model = self.ui.listView.model()
            
            if model is None or model.rowCount() == 0:
                QMessageBox.warning(self, "Warning", "No treatment data to save")
                return
            
            items = []
            for row in range(model.rowCount()):
                index = model.index(row, 0)
                items.append(model.data(index))
            
            with open("treatment_of_disease.txt", "w", encoding="utf-8") as f:
                for item in items:
                    f.write(item + "\n\n")
            
            QMessageBox.information(self, "Success", "Treatment data saved to 'treatment_of_disease.txt'")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save treatment: {str(e)}")

    def load_model(self):
        
        model = load_model('chaegaeg.keras', compile=False)
        self.model=model
        print("Model loaded successfully.")
        
        self.class_labels = ['COVID-19', 'Lung_opcaity', 'normal', 'Pneumonia']
    def open_detail_window(self):
        self.detail_window = DetailWidget(self.file_path,self)
        self.hide()
        self.detail_window.show()
 
    def resize_label(self, event):
        self.ui.label.setGeometry(self.ui.frame.rect())
        super().resizeEvent(event)
    
    def preprocess_image(self, img_path, target_size=(128, 128)):  

        try:
       
            img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')  
            
            
            img_array = image.img_to_array(img)  
            img_array = img_array / 255.0
          
            img_array = np.expand_dims(img_array, axis=0)  
            
            return img_array
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")
    def predict_image(self,img_path):
        img_array = self.preprocess_image(img_path)
        
        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = self.class_labels[predicted_class_index]
        confidence = np.max(predictions)
        
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

            if not file_path.lower().endswith(('.png', '.PNG')):
                QMessageBox.warning(self, "Invalid File", "Supported formats: JPG, JPEG")
                return
            self.file_path=file_path
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

            predicted_class, confidence= self.predict_image(file_path)
            self.start_scrapers(predicted_class)
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Prediction Result")
            msg_box.setText(f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}")
            
            msg_box.exec()

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
    def handle_scraped_data(self, data, scrape_type):
        model = QStringListModel()
        model.setStringList(data)

        if scrape_type == 'causes':
            self.ui.listView_2.setModel(model)
            self.ui.listView_2.setWordWrap(True)
        else:
            self.ui.listView.setModel(model)
            self.ui.listView.setWordWrap(True)
            
    def start_scrapers(self, predicted_class):
        if predicted_class == 'normal':
            return
        self.causes_worker = ScraperWorker(predicted_class, 'causes')
        self.causes_thread = QThread()
        self.causes_worker.moveToThread(self.causes_thread)
        self.causes_worker.finished.connect(self.handle_scraped_data)
        self.causes_thread.started.connect(self.causes_worker.run)
        self.causes_thread.start()

        self.treatment_worker = ScraperWorker(predicted_class, 'treatment')
        self.treatment_thread = QThread()
        self.treatment_worker.moveToThread(self.treatment_thread)
        self.treatment_worker.finished.connect(self.handle_scraped_data)
        self.treatment_thread.started.connect(self.treatment_worker.run)
        self.treatment_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.showMaximized()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"Application failed: {str(e)}")
        sys.exit(1)