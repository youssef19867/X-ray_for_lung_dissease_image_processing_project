from Detail_Page_UI import Ui_Form  
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget
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
class Histogram_operations:
    def __init__(self,pixmap,ui,parent):
        self.pixmap=pixmap
        self.ui=ui
        self.parent=parent
        self.isequalized=False
    
    def histogram_equalization(self, img2):
        def work(self, img2):
            try:
                img = img2
                histogram = {i: 0 for i in range(256)}
                H, W = img.shape
                for y in range(H):
                    for x in range(W):
                        intensity = img[y, x]
                        histogram[intensity] += 1
                self.ui.progressBar.setValue(25)  # 25% after histogram
                
                prob_histo = {i: 0 for i in range(256)}
                for nj in histogram.keys():
                    prob_histo[nj] = histogram[nj] / (H * W)
                self.ui.progressBar.setValue(50)  # 50% after prob_histo
                
                cp = {i: 0 for i in range(256)}
                total = 0
                for cumul in prob_histo.keys():
                    total += prob_histo[cumul]
                    cp[cumul] = total
                self.ui.progressBar.setValue(50)
                
                maxval = max(histogram.keys())
                self.ui.progressBar.setValue(75)  # 75% after finding maxval
                
                for key in cp.keys():
                    cp[key] = cp[key] * maxval
                
                new_img = np.zeros_like(img)
                for oldkey in histogram.keys():
                    progress = 75 + ((oldkey + 1) * 25 // 256)
                    self.ui.progressBar.setValue(progress)
                    for y in range(H):
                        for x in range(W):
                            if img[y, x] == oldkey:
                                new_img[y, x] = cp[oldkey]
                self.ui.progressBar.setValue(100)  # Complete to 100%
                self.ui.progressBar.setValue(self.ui.progressBar.minimum())  # Reset
                
                img2 = new_img
                self.parent.zamodified_img=new_img
                self.parent.display_image(new_img)
                print("done")
                return new_img
            except Exception as e:  
                traceback.print_exc()
                
        if self.isequalized==False:
            thread = threading.Thread(target=work, args=(self, img2,))
            thread.start()
            self.isequalized=True
        
class spatial_filtering:
    def __init__(self,pixmap,ui,parent):
        self.ui=ui
        self.pixmap=pixmap
        self.parent=parent
    def contrast_stretching(self,img):
        pass
    def log_transform(self,img):
        pass
    def gamma_correction(self, img):
        try:
            Lmax = 255
            r = max(0.1, self.ui.horizontalSlider.value() / 10.0)
            
            img_float = img.astype(np.float32)
            
            corrected_img = Lmax * (img_float / Lmax) ** r
        
            corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)
        
            self.parent.zamodified_img = corrected_img
            self.parent.display_image(corrected_img)
        except Exception as e:
            traceback.print_exc()
        

    def inverse_image(self,img):
        pass
    def image_thresholding(self,img):
        pass
class DetailWidget(QWidget):  # Changed from QMainWindow to QWidget
    def __init__(self,location_of_image,main_window):
        super().__init__()
        self.ui = Ui_Form()  # Changed to use Ui_Form instead of Ui_MainWindow
        self.ui.setupUi(self)
        self.Zamodified_img=cv2.imread(location_of_image,cv2.IMREAD_GRAYSCALE)
     
        self.display_image(self.Zamodified_img)
        self.main_window=main_window
        self.ui.toolButton.clicked.connect(self.back_to_main_window)
        self.histobj=Histogram_operations(self.Zamodified_img,self.ui,self)
        self.spatialobj=spatial_filtering(self.Zamodified_img,self.ui,self)
        self.ui.checkBox.toggled.connect(
        lambda checked: self.histobj.histogram_equalization(self.Zamodified_img) if checked else None)
        self.ui.horizontalSlider.valueChanged.connect(lambda : self.spatialobj.gamma_correction(self.Zamodified_img))
          

    def save_image(self):
        if not hasattr(self, 'Zamodified_img'):
            return
            
        file_path = "newimg.png"
        if file_path:
            cv2.imwrite(file_path, self.Zamodified_img)   
    def back_to_main_window(self):
        self.main_window.show()
        self.hide()
   

    
    def display_image(self, zamodified_img):
        try:
            # Set label size to match frame
            self.ui.label_11.setFixedSize(self.ui.frame.size())
            
            # Center-align the content within the label
            self.ui.label_11.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Handle different image types
            if isinstance(zamodified_img, QPixmap):
                # Scale QPixmap directly
                scaled_pixmap = zamodified_img.scaled(
                    self.ui.label_11.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.ui.label_11.setPixmap(scaled_pixmap)

            elif isinstance(zamodified_img, QImage):
                # Convert QImage to QPixmap and scale
                pixmap = QPixmap.fromImage(zamodified_img)
                scaled_pixmap = pixmap.scaled(
                    self.ui.label_11.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.ui.label_11.setPixmap(scaled_pixmap)

            elif isinstance(zamodified_img, np.ndarray):
                # Handle OpenCV image (numpy array)
                height, width = zamodified_img.shape[:2]
                
                # Convert BGR to RGB for color images
                if len(zamodified_img.shape) == 3 and zamodified_img.shape[2] == 3:
                    rgb_image = cv2.cvtColor(zamodified_img, cv2.COLOR_BGR2RGB)
                    bytes_per_line = 3 * width
                    q_format = QImage.Format.Format_RGB888
                    img_data = rgb_image.data
                else:
                    # Grayscale image
                    if len(zamodified_img.shape) == 2:
                        img_data = zamodified_img.data
                        bytes_per_line = width
                        q_format = QImage.Format.Format_Grayscale8
                    else:
                        raise ValueError("Unsupported numpy image format")

                # Create QImage and ensure data is contiguous
                img_data_contiguous = np.ascontiguousarray(img_data)
                q_img = QImage(
                    img_data_contiguous.data, 
                    width, 
                    height, 
                    bytes_per_line, 
                    q_format
                )

                # Convert to QPixmap and scale
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(
                    self.ui.label_11.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.ui.label_11.setPixmap(scaled_pixmap)
                print("displayed")

            else:
                print(f"Unsupported image type: {type(zamodified_img)}")

        except Exception as e:
            print(f"Error displaying image: {str(e)}")
            import traceback
            traceback.print_exc()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetailWidget()
    window.showMaximized()
    sys.exit(app.exec())
