from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ImageEnhancementWidget(object):
    def setupUi(self, ImageEnhancementWidget):
        ImageEnhancementWidget.setObjectName("ImageEnhancementWidget")

        ImageEnhancementWidget.setStyleSheet("/* Main Application Background */\n"
"QWidget {\n"
"    background-color: #1A1C1D; /* Almost black */\n"
"    color: #FFFFFF; /* White text */\n"
"    font-family: \"Consolas\", \"Monospace\";\n"
"    font-size: 14px;\n"
"}\n"
"\n"
"/* Buttons */\n"
"QPushButton {\n"
"    background-color: #5CE1E6; /* Cold cyan */\n"
"    color: #1A1C1D; /* Dark text on bright button */\n"
"    border: 1px solid #5CE1E6;\n"
"    border-radius: 6px;\n"
"    padding: 6px 12px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #8CE3F5; /* Light neon on hover */\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #00B4D8; /* Stronger cyan on press */\n"
"}\n"
"\n"
"/* Text Inputs */\n"
"QLineEdit, QTextEdit {\n"
"    background-color: #2E3233; /* Dark panel */\n"
"    color: #FFFFFF;\n"
"    border: 1px solid #5CE1E6;\n"
"    border-radius: 4px;\n"
"    padding: 5px;\n"
"}\n"
"\n"
"QLineEdit:focus, QTextEdit:focus {\n"
"    border: 1px solid #00B4D8; /* Glow cyan focus */\n"
"}\n"
"\n"
"/* Labels */\n"
"QLabel {\n"
"    color: #8CE3F5;\n"
"}\n"
"\n"
"/* Scrollbars */\n"
"QScrollBar:vertical, QScrollBar:horizontal {\n"
"    background-color: #2E3233;\n"
"}\n"
"\n"
"QScrollBar::handle:vertical, QScrollBar::handle:horizontal {\n"
"    background-color: #5CE1E6;\n"
"    border-radius: 6px;\n"
"}\n"
"\n"
"/* Menus */\n"
"QMenuBar {\n"
"    background-color: #1A1C1D;\n"
"    color: #FFFFFF;\n"
"}\n"
"\n"
"QMenuBar::item:selected {\n"
"    background-color: #00B4D8;\n"
"}\n"
"\n"
"QMenu {\n"
"    background-color: #2E3233;\n"
"    color: #FFFFFF;\n"
"    border: 1px solid #5CE1E6;\n"
"}\n"
"\n"
"QMenu::item:selected {\n"
"    background-color: #5CE1E6;\n"
"    color: #1A1C1D;\n"
"}\n"
"\n"
"/* Tabs */\n"
"QTabWidget::pane {\n"
"    background: #2E3233;\n"
"    border: 1px solid #5CE1E6;\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"    background: #2E3233;\n"
"    color: #8CE3F5;\n"
"    padding: 8px 12px;\n"
"    border: 1px solid #5CE1E6;\n"
"    border-bottom: none;\n"
"    border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"}\n"
"\n"
"QTabBar::tab:selected {\n"
"    background: #1A1C1D;\n"
"}\n"
"\n"
"QTabBar::tab:hover {\n"
"    background: #5CE1E6;\n"
"    color: #1A1C1D;\n"
"}\n"
"\n"
"/* Sliders */\n"
"QSlider::groove:horizontal {\n"
"    background: #5A595C;\n"
"    height: 6px;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background: #5CE1E6;\n"
"    width: 16px;\n"
"    height: 16px;\n"
"    margin: -5px 0;\n"
"    border-radius: 8px;\n"
"}\n"
"\n"
"/* Progress Bars */\n"
"QProgressBar {\n"
"    background-color: #2E3233;\n"
"    border: 1px solid #5CE1E6;\n"
"    border-radius: 4px;\n"
"    color: #FFFFFF;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: #5CE1E6;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"/* Group Boxes */\n"
"QGroupBox {\n"
"    border: 1px solid #5CE1E6;\n"
"    border-radius: 4px;\n"
"    margin-top: 10px;\n"
"    padding-top: 10px;\n"
"    color: #8CE3F5;\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    subcontrol-position: top left;\n"
"    padding: 0 5px;\n"
"    background-color: #1A1C1D;\n"
"}")
        
        # Main vertical layout with proper spacing for centering
        self.verticalLayout = QtWidgets.QVBoxLayout(ImageEnhancementWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(10)  # Normal spacing between elements
        
        # Top layout with back button
        self.topLayout = QtWidgets.QHBoxLayout()
        self.topLayout.setObjectName("topLayout")
        self.backButton = QtWidgets.QPushButton(parent=ImageEnhancementWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.backButton.sizePolicy().hasHeightForWidth())
        self.backButton.setSizePolicy(sizePolicy)
        self.backButton.setMinimumSize(QtCore.QSize(80, 30))
        self.backButton.setObjectName("backButton")
        self.topLayout.addWidget(self.backButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.topLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.topLayout)
        
        # Add top spacer to push content to center
        topSpacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(topSpacerItem)
        
        # Image frames layout - centered horizontally
        self.imageFramesLayout = QtWidgets.QHBoxLayout()
        self.imageFramesLayout.setObjectName("imageFramesLayout")
        self.imageFramesLayout.setSpacing(10)
        
        # Add horizontal spacer at the beginning for centering
        leftSpacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.imageFramesLayout.addItem(leftSpacerItem)
        
        # Original image layout
        self.originalImageLayout = QtWidgets.QVBoxLayout()
        self.originalImageLayout.setObjectName("originalImageLayout")
        self.originalImageLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        self.originalImageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.originalImageLabel.setObjectName("originalImageLabel")
        self.originalImageLayout.addWidget(self.originalImageLabel)
        self.originalImageFrame = QtWidgets.QFrame(parent=ImageEnhancementWidget)
        self.originalImageFrame.setFixedHeight(300)
        self.originalImageFrame.setMinimumWidth(200)
        self.originalImageFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.originalImageFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.originalImageFrame.setObjectName("originalImageFrame")
        self.originalImageLayout.addWidget(self.originalImageFrame)
        self.imageFramesLayout.addLayout(self.originalImageLayout)
        
        # Enhanced image layout
        self.enhancedImageLayout = QtWidgets.QVBoxLayout()
        self.enhancedImageLayout.setObjectName("enhancedImageLayout")
        self.enhancedImageLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        self.enhancedImageLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.enhancedImageLabel.setObjectName("enhancedImageLabel")
        self.enhancedImageLayout.addWidget(self.enhancedImageLabel)
        self.enhancedImageFrame = QtWidgets.QFrame(parent=ImageEnhancementWidget)
        self.enhancedImageFrame.setFixedHeight(300)
        self.enhancedImageFrame.setMinimumWidth(200)
        self.enhancedImageFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.enhancedImageFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.enhancedImageFrame.setObjectName("enhancedImageFrame")
        self.enhancedImageLayout.addWidget(self.enhancedImageFrame)
        self.imageFramesLayout.addLayout(self.enhancedImageLayout)
        
        # Area of importance layout
        self.areaOfImportanceLayout = QtWidgets.QVBoxLayout()
        self.areaOfImportanceLayout.setObjectName("areaOfImportanceLayout")
        self.areaOfImportanceLabel = QtWidgets.QLabel(parent=ImageEnhancementWidget)
        self.areaOfImportanceLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.areaOfImportanceLabel.setObjectName("areaOfImportanceLabel")
        self.areaOfImportanceLayout.addWidget(self.areaOfImportanceLabel)
        self.areaOfImportanceFrame = QtWidgets.QFrame(parent=ImageEnhancementWidget)
        self.areaOfImportanceFrame.setFixedHeight(300)
        self.areaOfImportanceFrame.setMinimumWidth(200)
        self.areaOfImportanceFrame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.areaOfImportanceFrame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.areaOfImportanceFrame.setObjectName("areaOfImportanceFrame")
        self.areaOfImportanceLayout.addWidget(self.areaOfImportanceFrame)
        self.imageFramesLayout.addLayout(self.areaOfImportanceLayout)
        
        # Add horizontal spacer at the end for centering
        rightSpacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.imageFramesLayout.addItem(rightSpacerItem)
        
        self.verticalLayout.addLayout(self.imageFramesLayout)
        
        # Progress bar
        self.progressBar = QtWidgets.QProgressBar(parent=ImageEnhancementWidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setContentsMargins(0, 10, 0, 0)
        self.verticalLayout.addWidget(self.progressBar)
        
        # Add bottom spacer to push content to center
        bottomSpacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout.addItem(bottomSpacerItem)
        
        # Bottom layout
        self.bottomLayout = QtWidgets.QHBoxLayout()
        self.bottomLayout.setObjectName("bottomLayout")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.bottomLayout.addItem(spacerItem4)
        self.verticalLayout.addLayout(self.bottomLayout)

        self.retranslateUi(ImageEnhancementWidget)
        QtCore.QMetaObject.connectSlotsByName(ImageEnhancementWidget)

    def retranslateUi(self, ImageEnhancementWidget):
        _translate = QtCore.QCoreApplication.translate
        ImageEnhancementWidget.setWindowTitle(_translate("ImageEnhancementWidget", "Image Enhancement"))
        self.backButton.setText(_translate("ImageEnhancementWidget", "Back"))
        self.originalImageLabel.setText(_translate("ImageEnhancementWidget", "Original Image"))
        self.enhancedImageLabel.setText(_translate("ImageEnhancementWidget", "Enhanced Image"))
        self.areaOfImportanceLabel.setText(_translate("ImageEnhancementWidget", "Area of Importance"))