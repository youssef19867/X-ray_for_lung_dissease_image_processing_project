
#for simplicity you can:
1. Download the complete project folder with pre-configured virtual environment also contains the models and images,just make sure you are running the venv:
   [Google Drive Link]https://drive.google.com/drive/folders/1ckN0hZ7gtha-C-X_vKH7bgIc8hfrnI0I?usp=sharing

#everything below is incase you dont want to download the full thing from google drive and also some extra details on the project

```markdown
# Medical Image Classification System

This application classifies medical images (X-rays) and provides disease information from Wikipedia.

## Prerequisites

- Python 3.9.13

## Setup Instructions

### Option 1: Manual Setup (Recommended for Developers)

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/MacOS:
     ```bash
     source venv/bin/activate
     ```

3. **Install required packages**:
   ```bash
   pip install numpy pillow matplotlib joblib requests beautifulsoup4 opencv-python keras PyQt6 scikit-learn tensorflow
   ```

4. **Verify installation**:
   ```bash
   pip list
   ```
5.models come sepretly so you have to download them from the same drive:
https://drive.google.com/drive/folders/1ckN0hZ7gtha-C-X_vKH7bgIc8hfrnI0I?usp=sharing
and then take out all the models and put them in the same directory or folder as the .py files or the code they must not be inside a seprate folder
### Option 2: Quick Setup (For Non-Developers)

1. Download the complete project folder with pre-configured virtual environment also contains the models:
   [Google Drive Link](https://drive.google.com/drive/folders/1ckN0hZ7gtha-C-X_vKH7bgIc8hfrnI0I?usp=sharing)

2. In VS Code:
   - Open the project folder
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Select "Python: Select Interpreter"
   - Choose the virtual environment from the downloaded folder (`venv`)

## Running the Application

1. Activate virtual environment (if not already active):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Required Python Packages

The system requires these specific packages:
```
PyQt6==6.4.2
numpy==1.23.5
pillow==9.5.0
matplotlib==3.7.1
joblib==1.2.0
requests==2.28.2
beautifulsoup4==4.11.2
opencv-python==4.7.0.72
keras==2.12.0
scikit-learn==1.2.2
tensorflow==2.12.0
```

## Features

- Medical image classification (X-rays)
- Disease information scraping from Wikipedia
- Image enhancement tools
- Detailed diagnostic reports

## Troubleshooting

If you encounter issues:
1. Verify Python version:
   ```bash
   python --version
   ```
2. Check virtual environment activation
3. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```


## Note

The pre-trained model files must be downloaded separately from the [Google Drive link]:
https://drive.google.com/drive/folders/1ckN0hZ7gtha-C-X_vKH7bgIc8hfrnI0I?usp=sharing

