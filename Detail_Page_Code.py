from Detail_Page_UI import Ui_Form  
from Custom_Page_Code import custom_filter_widget
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
                
                lut = np.zeros(256, dtype=img.dtype)
                for i in range(256):
                    self.ui.progressBar.setValue(int((i / 255) * 100))
                    lut[i] = cp[i]

                
                new_img = lut[img]
                #self.ui.progressBar.setValue(100)
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
    def adaptive_limited_equalization(self,img2):
        try:
            # Get the original image
            img = self.parent.Zamodified_img
            
            # Check if image is grayscale or color
            if len(img.shape) == 2:
                # For grayscale images
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized = clahe.apply(img)
            else:
                # For color images, convert to LAB color space
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                
                # Split the LAB channels
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to the L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_equalized = clahe.apply(l)
                
                # Merge the channels back
                lab_equalized = cv2.merge((l_equalized, a, b))
                
                # Convert back to BGR color space
                equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
            
        # Update the image
        # self.parent.Zamodified_img = equalized
            self.parent.display_image(equalized)
            self.ui.progressBar.setValue(100)
            
        except Exception as e:
            traceback.print_exc()
    def adaptive_histogram_equalization(self,img2):
        try:
            # Get the original image
            img = self.parent.Zamodified_img
            
            # Check if image is grayscale or color
            if len(img.shape) == 2:
                # For grayscale images
                # Create AHE object (without clip limit)
                clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(8, 8))
                equalized = clahe.apply(img)
            else:
                # For color images, convert to LAB color space
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                
                # Split the LAB channels
                l, a, b = cv2.split(lab)
                
                # Apply AHE to the L channel (no clip limit for pure AHE)
                clahe = cv2.createCLAHE(clipLimit=0.0, tileGridSize=(8, 8))
                l_equalized = clahe.apply(l)
                
                # Merge the channels back
                lab_equalized = cv2.merge((l_equalized, a, b))
                
                # Convert back to BGR color space
                equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
            
            # Update the image
           # self.parent.Zamodified_img = equalized
            self.parent.display_image(equalized)
            self.ui.progressBar.setValue(100)
            
        except Exception as e:
            traceback.print_exc()
        
    def histogram_shifting(self,img2):
        try:
            # Get the original image
            img = self.parent.Zamodified_img
            
            # Get the slider value for brightness adjustment
            shift_value = 80  # Assuming slider range 0-100, with 50 as neutral
            
            # Create a matrix of ones with the same shape as the image
            ones = np.ones(img.shape, dtype=np.uint8) * abs(shift_value)
            
            # Perform the shifting operation
            if shift_value >= 0:
                # Increase brightness (add values)
                shifted = cv2.add(img, ones)
            else:
                # Decrease brightness (subtract values)
                shifted = cv2.subtract(img, ones)
            
            # Update the image
            #self.parent.Zamodified_img = shifted
            self.parent.display_image(shifted)
            self.ui.progressBar.setValue(100)
            
        except Exception as e:
            traceback.print_exc()
    def quantile_based_equalization(self,img2):
        try:
            # Get the original image
            img = self.parent.Zamodified_img
            
            # Check if image is grayscale or color
            if len(img.shape) == 2:
                # For grayscale images
                # Calculate the cumulative distribution function (CDF)
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                cdf = hist.cumsum()
                
                # Normalize the CDF to the range [0, 255]
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                cdf_normalized = cdf_normalized.astype(np.uint8)
                
                # Use the normalized CDF as a lookup table for pixel values
                equalized = cdf_normalized[img]
            else:
                # For color images, process each channel separately
                # Convert to YCrCb color space (to preserve color while adjusting luminance)
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                
                # Calculate CDF for Y channel
                hist = cv2.calcHist([y], [0], None, [256], [0, 256])
                cdf = hist.cumsum()
                
                # Normalize the CDF
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                cdf_normalized = cdf_normalized.astype(np.uint8)
                
                # Apply equalization to Y channel
                y_equalized = cdf_normalized[y]
                
                # Merge channels back
                ycrcb_equalized = cv2.merge((y_equalized, cr, cb))
                
                # Convert back to BGR
                equalized = cv2.cvtColor(ycrcb_equalized, cv2.COLOR_YCrCb2BGR)
            
            # Update the image
            #self.parent.Zamodified_img = equalized
            self.parent.display_image(equalized)
            self.ui.progressBar.setValue(100)
            
        except Exception as e:
            traceback.print_exc()
class spatial_filtering:
    def __init__(self,pixmap,ui,parent):
        self.ui=ui
        self.pixmap=pixmap
        self.parent=parent
        self.original_img=self.parent.Zamodified_img
    def contrast_stretching(self,img):
        try:
            self.ui.progressBar.setValue(10)
            img=self.parent.Zamodified_img
            Lmax = 255
            new_min=int(self.ui.lineEdit_2.text())
            new_max=int(self.ui.lineEdit.text())
            
            img_float = img.astype(np.float32)
            
            corrected_img = img_float-0//Lmax*(new_max-new_min)+new_min
        
            corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)
            self.ui.progressBar.setValue(100)
            #self.parent.Zamodified_img = corrected_img
            self.parent.display_image(corrected_img)
            self.ui.progressBar.setValue(self.ui.progressBar.minimum())
        except Exception as e:
            traceback.print_exc()
    def log_transform(self):
          try:
            self.ui.progressBar.setValue(10)
            img=self.parent.Zamodified_img
            Lmax = 255
            c=self.ui.verticalSlider_2.value()
            
            img_float = img.astype(np.float32)
            
            corrected_img = c * np.log(1 + img_float)
        
            corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)
            self.ui.progressBar.setValue(100)
            #self.parent.Zamodified_img = corrected_img
            self.parent.display_image(corrected_img)
            self.ui.progressBar.setValue(self.ui.progressBar.minimum())
          except Exception as e:
            traceback.print_exc()
    def gamma_correction(self):
        try:
                        # Start progress bar
            self.ui.progressBar.setValue(10)

            # Get original image
            img = self.original_img

            # Make sure it's float for processing
            img_float = img.astype(np.float32)

            # Compute gamma value safely
            gamma = max(0.1, self.ui.horizontalSlider.value() / 20.0)

            # Apply gamma correction
            corrected_img = 255 * np.power(img_float / 255.0, gamma)

            # Clip values to valid range and convert back to uint8
            corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

            # Update progress bar
            self.ui.progressBar.setValue(100)

            # Display corrected image
            self.parent.display_image(corrected_img)

            # Reset progress bar
            self.ui.progressBar.setValue(self.ui.progressBar.minimum())

        except Exception as e:
            traceback.print_exc()
        

    def inverse_image(self,img):
        try:
            self.ui.progressBar.setValue(10)
            img=self.parent.Zamodified_img
            Lmax = 255
            
            
            img_float = img.astype(np.float32)
            
            corrected_img = Lmax-img_float
        
            corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)
            self.ui.progressBar.setValue(100)
            #self.parent.Zamodified_img = corrected_img
            self.parent.display_image(corrected_img)
            self.ui.progressBar.setValue(self.ui.progressBar.minimum())
        except Exception as e:
            traceback.print_exc()
    def image_thresholding(self):
        try:
            self.ui.progressBar.setValue(10)
            # Get current image from parent
            img = self.parent.Zamodified_img  # Lowercase z for consistency
            
            # Get threshold value from dial (0-255)
            threshold = self.ui.dial.value()
            
            # Vectorized thresholding using NumPy operations
            thresholded = np.where(img >= threshold, 255, 0).astype(np.uint8)
            
            self.ui.progressBar.setValue(90)
            
            # Update and display
            self.parent.zamodified_img = thresholded
            self.parent.display_image(thresholded)
            self.ui.progressBar.setValue(100)
            
        except Exception as e:
            traceback.print_exc()
        finally:
            self.ui.progressBar.setValue(self.ui.progressBar.minimum())
    
    def laplacian_sharpning(self):
        try:
            # Get the original image
            img = self.parent.Zamodified_img
            
            # Get the slider value (sharpening strength)
            strength = self.ui.horizontalSlider_2.value()
            
            # Apply Laplacian filter to detect edges
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            
            # Convert back to uint8
            laplacian = cv2.convertScaleAbs(laplacian)
            
            # Scale the effect based on slider value (assuming slider range is 0-100)
            # We'll scale it to a reasonable range, e.g., 0 to 2.0
            alpha = strength / 50.0  # This makes the middle value (50) equal to alpha=1.0
            
            # Apply sharpening: Original + (alpha * Laplacian)
            sharpened = cv2.addWeighted(img, 1.0, laplacian, alpha, 0)
            
            # Update the image
            #self.parent.Zamodified_img = sharpened
            self.parent.display_image(sharpened)
            self.ui.progressBar.setValue(100)
            
        except Exception as e:
            traceback.print_exc()
    def gaussian_blur(self):
        try:
            # Get the original image
            img = self.parent.Zamodified_img
            
            # Get the slider value for kernel size
            k_value = self.ui.verticalSlider.value()
            
            # Ensure k is odd (required for Gaussian blur kernel)
            k = k_value * 2 + 1 if k_value > 0 else 1
            
            # Apply Gaussian blur
            # The sigma value is left at 0 so OpenCV calculates it automatically based on kernel size
            blurred = cv2.GaussianBlur(img, (k, k), 0)
            
            # Update the image
            #self.parent.Zamodified_img = blurred
            self.parent.display_image(blurred)
            self.ui.progressBar.setValue(100)
            
        except Exception as e:
            traceback.print_exc()
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
        self.ui.checkBox_3.toggled.connect(
        lambda checked: self.histobj.adaptive_limited_equalization(self.Zamodified_img) if checked else None
        )
        self.ui.checkBox_2.toggled.connect(
        lambda checked: self.histobj.adaptive_histogram_equalization(self.Zamodified_img) if checked else None
        )
        self.ui.checkBox_4.toggled.connect(
        lambda checked: self.histobj.histogram_shifting(self.Zamodified_img) if checked else None
        )
        self.ui.checkBox_5.toggled.connect(
        lambda checked: self.histobj.quantile_based_equalization(self.Zamodified_img) if checked else None
        )
        self.ui.horizontalSlider.valueChanged.connect(self.spatialobj.gamma_correction)
        self.ui.horizontalSlider_2.valueChanged.connect(self.spatialobj.laplacian_sharpning)
        self.ui.verticalSlider_2.valueChanged.connect(self.spatialobj.log_transform)
        self.ui.verticalSlider.valueChanged.connect(self.spatialobj.gaussian_blur)
        self.ui.pushButton_2.clicked.connect(self.spatialobj.contrast_stretching)
        self.ui.pushButton_3.clicked.connect(self.spatialobj.inverse_image) 
        self.ui.dial.valueChanged.connect(self.spatialobj.image_thresholding)
        self.ui.pushButton_5.clicked.connect(self.open_custom_filter_page)
    def open_custom_filter_page(self):
        self.custom_filter_page = custom_filter_widget()
        self.hide()
        self.custom_filter_page.show()
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
