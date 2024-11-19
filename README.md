# MNIST Classification with CI/CD Pipeline

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with an automated CI/CD pipeline using GitHub Actions. The model is designed to achieve >95% accuracy in a single epoch while maintaining less than 25,000 parameters.

## Project Structure
```
├── model/
│ ├── init.py
│ └── network.py # CNN architecture
├── tests/
│ └── test_model.py # Model tests
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # CI/CD configuration
├── train.py # Training script
├── requirements.txt # Dependencies
└── .gitignore


## Model Architecture
- Input Layer: Accepts 28x28 grayscale images
- Conv Layer 1: 6 filters, 3x3 kernel
- Conv Layer 2: 12 filters, 3x3 kernel
- Fully Connected 1: 32 neurons
- Output Layer: 10 classes (digits 0-9)
- Total Parameters: < 25,000

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest
- tqdm

## Local Setup

1. Clone the repository:
```
bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate virtual environment:
```
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
bash
pip install -r requirements.txt
```

4. Train the model:
```
bash
python train.py
```

## Testing
Run the tests using:
```
bash
pytest tests/
```

The tests verify:
- Model parameter count (< 25,000)
- Input/output dimensions
- Model accuracy (> 95%)

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs CPU-only PyTorch
3. Trains the model
4. Runs all tests
5. Saves the trained model as an artifact

## Model Performance
- Training completed in 1 epoch
- Target accuracy: > 95%
- Parameter count: < 25,000
- Trained models are saved with timestamp and accuracy

## File Descriptions
- `model/network.py`: Defines the CNN architecture
- `train.py`: Contains training loop and evaluation code
- `tests/test_model.py`: Contains test cases
- `.github/workflows/ml-pipeline.yml`: Defines CI/CD pipeline
- `requirements.txt`: Lists project dependencies

## Notes
- Models and datasets are automatically excluded from git tracking
- CPU-only PyTorch is used in CI/CD pipeline
- Trained models are saved with accuracy metrics in filename

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License

Copyright (c) 2024 Abhijeet Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.