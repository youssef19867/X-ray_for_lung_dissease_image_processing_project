from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ImageEnhancementWidget(object):
    def setupUi(self, ImageEnhancementWidget):
        ImageEnhancementWidget.setObjectName("ImageEnhancementWidget")

        # Enhanced cyberpunk-inspired stylesheet with better visual hierarchy and depth
        ImageEnhancementWidget.setStyleSheet("""
        /* Main Application Background */
QWidget {
    background-color: #2D2A2E; /* Dark gray background */
    color: #FFD866; /* Golden text */
    font-family: "Consolas", "Monospace";
    font-size: 14px;
}

/* Buttons */
QPushButton {
    background-color: #FFD866; /* Golden hue */
    color: #2D2A2E; /* Dark gray text for contrast */
    border: 1px solid #FFD866; /* Golden border */
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 14px;
}

QPushButton:hover {
    background-color: #FFB52E; /* Slightly darker golden hue on hover */
    border: 1px solid #FFB52E;
}

QPushButton:pressed {
    background-color: #FF9F00; /* Even darker golden hue when pressed */
    border: 1px solid #FF9F00;
}

/* Text Inputs (QLineEdit, QTextEdit) */
QLineEdit, QTextEdit {
    background-color: #3E3D40; /* Dark gray */
    color: #FFD866; /* Golden text */
    border: 1px solid #5A595C; /* Slightly lighter gray border */
    border-radius: 4px;
    padding: 5px;
    font-size: 14px;
}
/* Vertical Sliders (matching horizontal style) */
QSlider::groove:vertical {
    background-color: #5A595C;
    width: 6px;  /* Changed from height for vertical */
    border-radius: 3px;
}

QSlider::handle:vertical {
    background-color: #FFD866;
    width: 16px;  /* Maintain same size as horizontal */
    height: 16px;
    margin: 0 -5px;  /* Adjusted for vertical orientation */
    border-radius: 8px;
}

/* Optional: Add these if you want to style tick marks */
QSlider::sub-page:vertical {
    background-color: #FFD866;
}

QSlider::add-page:vertical {
    background-color: #5A595C;
}

QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #FFD866; /* Yellow border on focus */
}

/* Labels */
QLabel {
    color: #FFD866; /* Golden text */
    font-size: 14px;
}

/* Scrollbars */
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #2D2A2E; /* Dark gray */
}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #5A595C; /* Lighter gray */
    border-radius: 6px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

/* Menus */
QMenuBar {
    background-color: #2D2A2E; /* Dark gray */
    color: #FFD866; /* Golden text */
}

QMenuBar::item {
    background-color: transparent;
    padding: 5px 10px;
}

QMenuBar::item:selected {
    background-color: #5A595C; /* Lighter gray */
}

QMenu {
    background-color: #3E3D40; /* Dark gray */
    color: #FFD866; /* Golden text */
    border: 1px solid #5A595C;
}

QMenu::item {
    padding: 5px 20px;
}

QMenu::item:selected {
    background-color: #5A595C;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #5A595C;
    background-color: #3E3D40;
}

QTabBar::tab {
    background-color: #3E3D40;
    color: #FFD866; /* Golden text */
    padding: 8px 12px;
    border: 1px solid #5A595C;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #2D2A2E;
    border-color: #5A595C;
}

QTabBar::tab:hover {
    background-color: #FFD866;
    color: #2D2A2E;
}

/* Sliders */
QSlider::groove:horizontal {
    background-color: #5A595C;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #FFD866;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

/* Progress Bars */
QProgressBar {
    background-color: #3E3D40;
    border: 1px solid #5A595C;
    border-radius: 4px;
    text-align: center;
    color: #FFD866; /* Golden text */
}

QProgressBar::chunk {
    background-color: #FFD866;
    border-radius: 4px;
}

/* Checkboxes and Radio Buttons */
QCheckBox, QRadioButton {
    color: #FFD866; /* Golden text */
    spacing: 5px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #5A595C;
    border-radius: 3px;
    background-color: #3E3D40;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #FFD866;
    border: 1px solid #FFD866;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #5A595C;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
    color: #FFD866; /* Golden text */
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    background-color: #2D2A2E;
}
        """)
        
        # Main vertical layout with proper spacing for centering
        self.verticalLayout = QtWidgets.QVBoxLayout(ImageEnhancementWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(20, 20, 20, 20)  # Increased margins
        self.verticalLayout.setSpacing(15)  # Increased spacing for better visual hierarchy
        
        # Top layout with back button and title
        self.topLayout = QtWidgets.QHBoxLayout()
        self.topLayout.setObjectName("topLayout")
        
        # Back button with icon
        self.backButton = QtWidgets.QPushButton(parent=ImageEnhancementWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.backButton.sizePolicy().hasHeightForWidth())
        self.backButton.setSizePolicy(sizePolicy)
        self.backButton.setMinimumSize(QtCore.QSize(100, 36))  # Larger button
        self.backButton.setObjectName("backButton")
        # Add back icon if available
        try:
            self.backButton.setIcon(QtGui.QIcon.fromTheme("go-previous"))
            self.backButton.setIconSize(QtCore.QSize(16, 16))
        except:
            pass  # If icon not available, continue without it
        
        self.topLayout.addWidget(self.backButton)
        
        # Title label - centered with large font
        self.titleLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.topLayout.addWidget(self.titleLabel)
        
        # Empty widget for layout balance
        spacerItem = QtWidgets.QSpacerItem(100, 36, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.topLayout.addItem(spacerItem)
        
        self.verticalLayout.addLayout(self.topLayout)
        
        # Add separator line
        self.separator = QtWidgets.QFrame(parent=ImageEnhancementWidget)
        self.separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.separator.setObjectName("separator")
        self.separator.setStyleSheet("background-color: #00E5FF; margin: 5px 0;")
        self.separator.setMaximumHeight(2)
        self.verticalLayout.addWidget(self.separator)
        
        # Add top spacer to push content to center
        topSpacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(topSpacerItem)
        
        # Image frames layout - centered horizontally
        self.imageFramesLayout = QtWidgets.QHBoxLayout()
        self.imageFramesLayout.setObjectName("imageFramesLayout")
        self.imageFramesLayout.setSpacing(20)  # Increased spacing between images
        
        # Add horizontal spacer at the beginning for centering
        leftSpacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.imageFramesLayout.addItem(leftSpacerItem)
        
        # Original image layout
        self.originalImageLayout = QtWidgets.QVBoxLayout()
        self.originalImageLayout.setObjectName("originalImageLayout")
        self.originalImageLayout.setSpacing(10)  # Spacing between label and image
        
        self.originalImageLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        self.originalImageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.originalImageLabel.setObjectName("originalImageLabel")
        # Style the label
        font = QtGui.QFont()
        font.setBold(True)
        self.originalImageLabel.setFont(font)
        self.originalImageLayout.addWidget(self.originalImageLabel)
        
        # Frame for original image with improved styling
        self.originalImageFrame = QtWidgets.QFrame(parent=ImageEnhancementWidget)
        self.originalImageFrame.setFixedSize(QtCore.QSize(300, 300))  # Square frame
        self.originalImageFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.originalImageFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.originalImageFrame.setObjectName("originalImageFrame")
        
        # Add image placeholder
        self.originalImageFrameLayout = QtWidgets.QVBoxLayout(self.originalImageFrame)
        self.originalImageDisplayLabel = QtWidgets.QLabel(parent=self.originalImageFrame)
        self.originalImageDisplayLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.originalImageDisplayLabel.setText("No Image")
        self.originalImageFrameLayout.addWidget(self.originalImageDisplayLabel)
        
        self.originalImageLayout.addWidget(self.originalImageFrame)
        self.imageFramesLayout.addLayout(self.originalImageLayout)
        
        # Enhanced image layout
        self.enhancedImageLayout = QtWidgets.QVBoxLayout()
        self.enhancedImageLayout.setObjectName("enhancedImageLayout")
        self.enhancedImageLayout.setSpacing(10)
        
        self.enhancedImageLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        self.enhancedImageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.enhancedImageLabel.setObjectName("enhancedImageLabel")
        # Style the label
        self.enhancedImageLabel.setFont(font)
        self.enhancedImageLayout.addWidget(self.enhancedImageLabel)
        
        # Frame for enhanced image with improved styling
        self.enhancedImageFrame = QtWidgets.QFrame(parent=ImageEnhancementWidget)
        self.enhancedImageFrame.setFixedSize(QtCore.QSize(300, 300))  # Square frame
        self.enhancedImageFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.enhancedImageFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.enhancedImageFrame.setObjectName("enhancedImageFrame")
        
        # Add image placeholder
        self.enhancedImageFrameLayout = QtWidgets.QVBoxLayout(self.enhancedImageFrame)
        self.enhancedImageDisplayLabel = QtWidgets.QLabel(parent=self.enhancedImageFrame)
        self.enhancedImageDisplayLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.enhancedImageDisplayLabel.setText("Processing...")
        self.enhancedImageFrameLayout.addWidget(self.enhancedImageDisplayLabel)
        
        self.enhancedImageLayout.addWidget(self.enhancedImageFrame)
        self.imageFramesLayout.addLayout(self.enhancedImageLayout)
        
        # Area of importance layout
        self.areaOfImportanceLayout = QtWidgets.QVBoxLayout()
        self.areaOfImportanceLayout.setObjectName("areaOfImportanceLayout")
        self.areaOfImportanceLayout.setSpacing(10)
        
        self.areaOfImportanceLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        self.areaOfImportanceLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.areaOfImportanceLabel.setObjectName("areaOfImportanceLabel")
        # Style the label
        self.areaOfImportanceLabel.setFont(font)
        self.areaOfImportanceLayout.addWidget(self.areaOfImportanceLabel)
        
        # Frame for area of importance with improved styling
        self.areaOfImportanceFrame = QtWidgets.QFrame(parent=ImageEnhancementWidget)
        self.areaOfImportanceFrame.setFixedSize(QtCore.QSize(300, 300))  # Square frame
        self.areaOfImportanceFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.areaOfImportanceFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.areaOfImportanceFrame.setObjectName("areaOfImportanceFrame")
        
        # Add image placeholder
        self.areaOfImportanceFrameLayout = QtWidgets.QVBoxLayout(self.areaOfImportanceFrame)
        self.areaOfImportanceDisplayLabel = QtWidgets.QLabel(parent=self.areaOfImportanceFrame)
        self.areaOfImportanceDisplayLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.areaOfImportanceDisplayLabel.setText("Select Area")
        self.areaOfImportanceFrameLayout.addWidget(self.areaOfImportanceDisplayLabel)
        
        self.areaOfImportanceLayout.addWidget(self.areaOfImportanceFrame)
        self.imageFramesLayout.addLayout(self.areaOfImportanceLayout)
        
        # Add horizontal spacer at the end for centering
        rightSpacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.imageFramesLayout.addItem(rightSpacerItem)
        
        self.verticalLayout.addLayout(self.imageFramesLayout)
        
        # Actions layout (buttons)
        self.actionsLayout = QtWidgets.QHBoxLayout()
        self.actionsLayout.setObjectName("actionsLayout")
        self.actionsLayout.setSpacing(15)
        
        # Center the buttons
        spacerActionLeft = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.actionsLayout.addItem(spacerActionLeft)
        
        # Load Image button
        self.loadImageButton = QtWidgets.QPushButton(parent=ImageEnhancementWidget)
        self.loadImageButton.setObjectName("loadImageButton")
        self.loadImageButton.setMinimumSize(QtCore.QSize(150, 40))
        self.actionsLayout.addWidget(self.loadImageButton)
        
        # Enhance button
        self.enhanceButton = QtWidgets.QPushButton(parent=ImageEnhancementWidget)
        self.enhanceButton.setObjectName("enhanceButton")
        self.enhanceButton.setMinimumSize(QtCore.QSize(150, 40))
        self.actionsLayout.addWidget(self.enhanceButton)
        
        # Save button
        self.saveButton = QtWidgets.QPushButton(parent=ImageEnhancementWidget)
        self.saveButton.setObjectName("saveButton")
        self.saveButton.setMinimumSize(QtCore.QSize(150, 40))
        self.actionsLayout.addWidget(self.saveButton)
        
        spacerActionRight = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.actionsLayout.addItem(spacerActionRight)
        
        self.verticalLayout.addLayout(self.actionsLayout)
        
        # Settings group box
        self.settingsGroupBox = QtWidgets.QGroupBox(parent=ImageEnhancementWidget)
        self.settingsGroupBox.setObjectName("settingsGroupBox")
        self.settingsGroupBox.setMinimumHeight(150)
        
        # Layout for settings
        self.settingsLayout = QtWidgets.QGridLayout(self.settingsGroupBox)
        self.settingsLayout.setObjectName("settingsLayout")
        self.settingsLayout.setContentsMargins(15, 15, 15, 15)
        self.settingsLayout.setSpacing(10)
        
        # Brightness slider
        self.brightnessLabel = QtWidgets.QLabel(parent=self.settingsGroupBox)
        self.brightnessLabel.setObjectName("brightnessLabel")
        self.settingsLayout.addWidget(self.brightnessLabel, 0, 0)
        
        self.brightnessSlider = QtWidgets.QSlider(parent=self.settingsGroupBox)
        self.brightnessSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.brightnessSlider.setRange(-100, 100)
        self.brightnessSlider.setValue(0)
        self.brightnessSlider.setObjectName("brightnessSlider")
        self.settingsLayout.addWidget(self.brightnessSlider, 0, 1)
        
        self.brightnessValue = QtWidgets.QLabel(parent=self.settingsGroupBox)
        self.brightnessValue.setObjectName("brightnessValue")
        self.brightnessValue.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.settingsLayout.addWidget(self.brightnessValue, 0, 2)
        
        # Contrast slider
        self.contrastLabel = QtWidgets.QLabel(parent=self.settingsGroupBox)
        self.contrastLabel.setObjectName("contrastLabel")
        self.settingsLayout.addWidget(self.contrastLabel, 1, 0)
        
        self.contrastSlider = QtWidgets.QSlider(parent=self.settingsGroupBox)
        self.contrastSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.contrastSlider.setRange(-100, 100)
        self.contrastSlider.setValue(0)
        self.contrastSlider.setObjectName("contrastSlider")
        self.settingsLayout.addWidget(self.contrastSlider, 1, 1)
        
        self.contrastValue = QtWidgets.QLabel(parent=self.settingsGroupBox)
        self.contrastValue.setObjectName("contrastValue")
        self.contrastValue.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.settingsLayout.addWidget(self.contrastValue, 1, 2)
        
        # Saturation slider
        self.saturationLabel = QtWidgets.QLabel(parent=self.settingsGroupBox)
        self.saturationLabel.setObjectName("saturationLabel")
        self.settingsLayout.addWidget(self.saturationLabel, 2, 0)
        
        self.saturationSlider = QtWidgets.QSlider(parent=self.settingsGroupBox)
        self.saturationSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.saturationSlider.setRange(-100, 100)
        self.saturationSlider.setValue(0)
        self.saturationSlider.setObjectName("saturationSlider")
        self.settingsLayout.addWidget(self.saturationSlider, 2, 1)
        
        self.saturationValue = QtWidgets.QLabel(parent=self.settingsGroupBox)
        self.saturationValue.setObjectName("saturationValue")
        self.saturationValue.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.settingsLayout.addWidget(self.saturationValue, 2, 2)
        
        # Add settings to main layout
        self.verticalLayout.addWidget(self.settingsGroupBox)
        
        # Progress bar with nice styling
        self.progressBar = QtWidgets.QProgressBar(parent=ImageEnhancementWidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setFixedHeight(16)  # Slim progress bar
        self.progressBar.setTextVisible(True)
        self.verticalLayout.addWidget(self.progressBar)
        
        # Status label
        self.statusLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        self.statusLabel.setObjectName("statusLabel")
        self.statusLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = QtGui.QFont()
        font.setItalic(True)
        self.statusLabel.setFont(font)
        self.verticalLayout.addWidget(self.statusLabel)
        
        # Add bottom spacer to push content to center
        bottomSpacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(bottomSpacerItem)

        self.retranslateUi(ImageEnhancementWidget)
        QtCore.QMetaObject.connectSlotsByName(ImageEnhancementWidget)
        
        # Connect signals and slots for live updates
        self.brightnessSlider.valueChanged.connect(lambda value: self.brightnessValue.setText(f"{value}%"))
        self.contrastSlider.valueChanged.connect(lambda value: self.contrastValue.setText(f"{value}%"))
        self.saturationSlider.valueChanged.connect(lambda value: self.saturationValue.setText(f"{value}%"))

    def retranslateUi(self, ImageEnhancementWidget):
        _translate = QtCore.QCoreApplication.translate
        ImageEnhancementWidget.setWindowTitle(_translate("ImageEnhancementWidget", "Knn Enhancement - AI Image Enhancement"))
        self.backButton.setText(_translate("ImageEnhancementWidget", "< Back"))
        self.titleLabel.setText(_translate("ImageEnhancementWidget", "IMAGE ENHANCER"))
        self.originalImageLabel.setText(_translate("ImageEnhancementWidget", "ORIGINAL"))
        self.enhancedImageLabel.setText(_translate("ImageEnhancementWidget", "ENHANCED"))
        self.areaOfImportanceLabel.setText(_translate("ImageEnhancementWidget", "FOCUS AREA"))
        self.loadImageButton.setText(_translate("ImageEnhancementWidget", "Load Image"))
        self.enhanceButton.setText(_translate("ImageEnhancementWidget", "Enhance"))
        self.saveButton.setText(_translate("ImageEnhancementWidget", "Save Result"))
        self.settingsGroupBox.setTitle(_translate("ImageEnhancementWidget", "Enhancement Settings"))
        self.brightnessLabel.setText(_translate("ImageEnhancementWidget", "Brightness:"))
        self.brightnessValue.setText(_translate("ImageEnhancementWidget", "0%"))
        self.contrastLabel.setText(_translate("ImageEnhancementWidget", "Contrast:"))
        self.contrastValue.setText(_translate("ImageEnhancementWidget", "0%"))
        self.saturationLabel.setText(_translate("ImageEnhancementWidget", "Saturation:"))
        self.saturationValue.setText(_translate("ImageEnhancementWidget", "0%"))
        self.statusLabel.setText(_translate("ImageEnhancementWidget", "Ready to enhance. Load an image to begin."))