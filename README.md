# Deep Learning Assignment 3: CNN for Image Classification

This project implements Convolutional Neural Networks (CNN) for image classification on MNIST and CIFAR-10 datasets using TensorFlow/Keras.

## Project Overview

This project implements CNN models for two datasets:
- **MNIST**: Handwritten digit classification (0-9)
- **CIFAR-10**: Object classification (10 categories)

The assignment covers:
1. Data preprocessing and train/test splitting
2. CNN model implementation with convolutional, pooling, and fully connected layers
3. Model training with appropriate loss functions and optimizers
4. Model evaluation with accuracy, precision, recall, and F1-score
5. Hyperparameter tuning experiments

## Requirements

- Python 3.12+
- uv (Python package manager)

## Installation

This project uses `uv` as the package manager. Follow these steps to set up the environment:

1. **Install uv** (if not already installed):
   ```bash
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd deeplearning-assignment-3
   ```

3. **Sync dependencies**:
   ```bash
   uv sync
   ```

   This will automatically:
   - Create a virtual environment (`.venv`)
   - Install all required packages (TensorFlow, matplotlib, scikit-learn)

## Project Structure

```
deeplearning-assignment-3/
├── abdullah_54.ipynb          # Main notebook with MNIST and CIFAR-10 implementations
├── CNN_MNIST_Report.md        # Detailed technical report for both datasets
├── assignment_doc.md          # Complete assignment documentation
├── README.md                  # This file (setup and usage guide)
├── pyproject.toml             # Project dependencies and configuration
└── .venv/                     # Virtual environment (created by uv)
```

## Usage

### Running the Notebook

1. **Activate the virtual environment**:
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook abdullah_54.ipynb
   ```
   
   Or if using VS Code, simply open the notebook and select the `.venv` kernel.

3. **Run all cells** in order:
   - **Part 1 (MNIST)**: Cells 1-7 for MNIST implementation
   - **Part 2 (CIFAR-10)**: Remaining cells for CIFAR-10 implementation
   
   Each part includes:
   - Download and preprocess the dataset
   - Build and train the CNN model
   - Evaluate performance with required metrics
   - Experiment with different hyperparameters

### Expected Results

**MNIST:**
- Training accuracy: ~98-99%
- Test accuracy: ~98.5%
- Training time: ~5-10 minutes

**CIFAR-10:**
- Training accuracy: ~75-85%
- Test accuracy: ~70-80%
- Training time: ~15-30 minutes (deeper architecture)

## Dependencies

The project uses the following main packages (managed by uv):

- `tensorflow>=2.20.0` - Deep learning framework
- `matplotlib>=3.10.8` - Visualization
- `scikit-learn>=1.8.0` - Evaluation metrics
- `seaborn>=0.13.2` - Additional plotting (optional)

All dependencies are specified in `pyproject.toml`.

## Model Architectures

### MNIST Model
- **Input**: 28×28×1 grayscale images
- **Convolutional layers**: 3 Conv2D layers (32, 64, 64 filters)
- **Pooling layers**: 2 MaxPooling2D layers
- **Fully connected layers**: Dense layer (64 units) + output layer (10 units)
- **Activation functions**: ReLU for hidden layers, Softmax for output

### CIFAR-10 Model
- **Input**: 32×32×3 color images
- **Convolutional layers**: 6 Conv2D layers (32, 32, 64, 64, 128, 128 filters)
- **Pooling layers**: 3 MaxPooling2D layers
- **Dropout layers**: Progressive dropout (0.2, 0.3, 0.4, 0.5)
- **Fully connected layers**: Dense layer (128 units) + output layer (10 units)
- **Activation functions**: ReLU for hidden layers, Softmax for output

## Training Configuration

### MNIST
- **Loss function**: Categorical cross-entropy
- **Optimizer**: Adam
- **Learning rate**: 0.001
- **Batch size**: 128
- **Epochs**: 10

### CIFAR-10
- **Loss function**: Categorical cross-entropy
- **Optimizer**: Adam
- **Learning rate**: 0.001
- **Batch size**: 64
- **Epochs**: 20

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)

## Hyperparameter Tuning

### MNIST Experiments
- Different learning rates (0.001, 0.0005)
- Different batch sizes (128, 64)

### CIFAR-10 Experiments
- Different learning rates (0.001, 0.0005)
- Different batch sizes (64, 32)

## Reports

Two comprehensive documentation files are included:

### 1. Technical Report (`CNN_MNIST_Report.md`)
Detailed technical report covering:

**Part 1 - MNIST:**
- Data preprocessing approach
- Model architecture design choices
- Training process and results
- Evaluation metrics and analysis
- Hyperparameter tuning findings

**Part 2 - CIFAR-10:**
- Deeper architecture justification
- Color image processing approach
- Dropout and regularization strategies
- Performance comparison with MNIST
- Key learnings from both datasets

### 2. Assignment Documentation (`assignment_doc.md`)
Complete assignment documentation including:
- Assignment overview and requirements
- Implementation structure
- Detailed explanation of both MNIST and CIFAR-10 implementations
- Step-by-step preprocessing and architecture details
- Comparison and analysis
- How to run instructions
- Results summary and conclusion

## Managing Dependencies with uv

### Add a new package
```bash
uv add package-name
```

### Remove a package
```bash
uv remove package-name
```

### Update all packages
```bash
uv sync --upgrade
```

### Show installed packages
```bash
uv pip list
```

## Troubleshooting

### Issue: TensorFlow not found
```bash
uv sync --reinstall
```

### Issue: Jupyter kernel not found
```bash
uv add jupyter ipykernel
python -m ipykernel install --user --name=.venv
```


