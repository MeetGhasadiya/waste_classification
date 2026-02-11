# ♻️ Waste Classification using Deep Learning

A deep learning-powered web application that classifies waste into 10 different categories using MobileNetV2 architecture. This project helps promote proper waste disposal and recycling practices through AI-driven image classification.

## 🎯 Project Overview

This project uses transfer learning with MobileNetV2 (pre-trained on ImageNet) to classify waste images into 10 distinct categories. The model is deployed as an interactive web application using Streamlit, allowing users to upload images and receive instant classification results.

## 📊 Dataset

- **Source**: Garbage Classification V2 from Kaggle
- **Total Images**: Standardized 256x256 images
- **Split Ratio**: 
  - Training: 70%
  - Validation: 15%
  - Test: 15%

## 🗂️ Waste Categories

The model classifies waste into the following 10 categories:

1. **Battery** 🔋 - Electronic waste requiring special disposal
2. **Biological** 🌱 - Organic waste suitable for composting
3. **Cardboard** 📦 - Recyclable paper product
4. **Clothes** 👕 - Textile waste
5. **Glass** 🍾 - Recyclable glass materials
6. **Metal** 🔩 - Recyclable metal items
7. **Paper** 📄 - Recyclable paper products
8. **Plastic** ♻️ - Various plastic items
9. **Shoes** 👟 - Footwear waste
10. **Trash** 🗑️ - General non-recyclable waste

## 🧠 Model Architecture

### Base Model
- **Architecture**: MobileNetV2 (ImageNet pre-trained)
- **Input Shape**: 224x224x3
- **Transfer Learning Strategy**: Two-phase training

### Classification Head
```
MobileNetV2 (frozen) 
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (10, softmax)
```

### Training Strategy

**Phase 1: Feature Extraction**
- Backbone frozen
- Train only classification head
- Epochs: 6
- Learning rate: 1e-4
- Optimizer: Adam

**Phase 2: Fine-Tuning**
- Unfroze last 30 layers
- Fine-tune with lower learning rate
- Epochs: 6
- Early stopping with patience=3

### Data Augmentation
- Rescaling: 1/255
- Rotation: ±20°
- Horizontal flip: Yes
- Zoom range: 0.2

## 📈 Model Performance

The model achieves high accuracy on the test set with detailed metrics available through the classification report and confusion matrix in the training notebook.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   cd e:\Pythom Projects\waste_classification
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model file**
   - Ensure `waste_model.h5` exists in the project directory

## 💻 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. Click "Browse files" or drag and drop an image
2. Upload a waste image (JPG, JPEG, or PNG format)
3. View the classification result with confidence score
4. Get instant feedback on waste type

## 📁 Project Structure

```
waste_classification/
├── app.py                  # Streamlit web application
├── waste_model.h5          # Trained model (H5 format)
├── best_model.keras        # Model checkpoint (Keras format)
├── waste1.ipynb           # Training notebook
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
└── venv/                  # Virtual environment (not tracked)
```

## 🔧 Technical Details

### Model Training
- **Framework**: TensorFlow/Keras
- **Callbacks**:
  - ModelCheckpoint (save best model)
  - EarlyStopping (prevent overfitting)
  - ReduceLROnPlateau (adaptive learning rate)

### Deployment
- **Framework**: Streamlit
- **Image Processing**: PIL (Pillow)
- **Model Loading**: Cached with `@st.cache_resource`

### Requirements
```
streamlit==1.38.0
tensorflow==2.20.0
numpy
Pillow
```

## 📊 Model Training Notebook

The `waste1.ipynb` notebook contains:
- Dataset download from Kaggle
- Data preprocessing and splitting
- Model architecture definition
- Two-phase training process
- Model evaluation
- Classification report
- Confusion matrix visualization

## 🎨 Features

- ✅ Real-time waste classification
- ✅ User-friendly web interface
- ✅ Instant predictions with confidence scores
- ✅ Support for common image formats
- ✅ Lightweight MobileNetV2 architecture
- ✅ High accuracy across all waste categories

## 🔍 Model Evaluation

The trained model is evaluated using:
- **Test accuracy** metrics
- **Classification report** (precision, recall, F1-score)
- **Confusion matrix** visualization
- Per-class performance analysis

## 🌍 Environmental Impact

Proper waste classification contributes to:
- ♻️ Improved recycling rates
- 🌱 Reduced landfill waste
- 🌍 Environmental conservation
- 💡 Promoting recycling awareness
- 🔄 Supporting circular economy

## 🛠️ Development

### Adding New Features
- Modify `app.py` for UI changes
- Update model in `waste1.ipynb` for improvements
- Retrain and export new model as `waste_model.h5`

### Training Custom Model
1. Open `waste1.ipynb` in Jupyter/Colab
2. Download dataset from Kaggle
3. Follow the training pipeline
4. Export model and update in project directory

## 📝 License

This project is for educational and environmental awareness purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Made with ♻️ for a sustainable future**