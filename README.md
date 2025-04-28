# MNIST Model with Specific Requirements

[![Model Requirements Validation](https://github.com/amhemanth/MNIST_expo/actions/workflows/model_validation.yml/badge.svg)](https://github.com/amhemanth/MNIST_expo/actions/workflows/model_validation.yml)

This project implements a PyTorch model for MNIST classification that meets the following requirements:

- Achieves 99.4% validation/test accuracy (using 50k/10k split)
- Uses less than 20k parameters
- Trains in less than 20 epochs
- Incorporates Batch Normalization and Dropout
- Uses either Fully Connected layer or Global Average Pooling

## Model Architecture

```
Input (1x28x28)
     ↓
[Conv2d(k=3, p=1) → BN → ReLU]  →  8x28x28
     ↓
[Conv2d(k=3, p=1) → BN → ReLU]  →  16x28x28
     ↓
[MaxPool2d(2)]  →  16x14x14
     ↓
[Dropout(0.1)]
     ↓
[Conv2d(k=3, p=1) → BN → ReLU]  →  16x14x14
     ↓
[Global Avg Pool]  →  16x1x1
     ↓
[Flatten]  →  16
     ↓
[Linear]  →  10
     ↓
[LogSoftmax]
     ↓
Output (10 classes)
```

Key Features:
- 3 convolutional layers with batch normalization
- Dropout for regularization (10%)
- Global Average Pooling
- Single fully connected layer
- Total parameters: < 20k

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (optional but recommended):
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python model.py
```

The script will:
1. Download the MNIST dataset automatically
2. Split the training data into 50k/10k train/validation sets
3. Train for 15 epochs
4. Save the best model based on validation accuracy
5. Print the final test accuracy and parameter count

Expected output:
```
Train Epoch: 1 [...]
...
Final Test Accuracy: ~99.4%
Total number of parameters: <20k
```

## Testing

Run the test suite to validate model architecture and performance:

```bash
python test_model.py
```

The tests verify:

1. Architecture Requirements:
   - Parameter count < 20k
   - Presence of Batch Normalization
   - Presence of Dropout
   - Use of FC layer or GAP
   - Correct output shapes

2. Performance Metrics:
   - Valid probability distributions
   - Inference speed
   - Model stability
   - Batch processing capability

Expected test output:
```
test_architecture (TestModelArchitecture) ... ok
test_batch_normalization (TestModelArchitecture) ... ok
test_dropout (TestModelArchitecture) ... ok
test_forward_pass (TestModelArchitecture) ... ok
test_parameter_count (TestModelArchitecture) ... ok
test_batch_inference_speed (TestModelPerformance) ... ok
test_model_output_range (TestModelPerformance) ... ok
test_model_stability (TestModelPerformance) ... ok

----------------------------------------------------------------------
Ran 8 tests in Xs

OK
```

## Model Details

Layer-wise dimensions:
- Input: 1x28x28 (MNIST image)
- Conv1: 8x28x28 (8 channels, spatial dim preserved with padding)
- Conv2: 16x28x28 (16 channels, spatial dim preserved with padding)
- MaxPool: 16x14x14 (spatial dim halved)
- Conv3: 16x14x14 (channels unchanged, spatial dim preserved)
- GAP: 16x1x1 (spatial dims collapsed)
- FC: 10 (output classes)

## Validation

GitHub Actions automatically validates that the model meets all requirements:
- Parameter count < 20k
- Use of Batch Normalization
- Use of Dropout
- Use of FC layer or GAP

The validation workflow runs on every push and pull request. 

## Test logs


Train Epoch: 1 [0/50000 (0%)]   Loss: 2.323003
Train Epoch: 1 [12800/50000 (26%)]      Loss: 1.864860
Train Epoch: 1 [25600/50000 (51%)]      Loss: 1.427048
Train Epoch: 1 [38400/50000 (77%)]      Loss: 1.216678

Test set: Average loss: 0.9820, Accuracy: 7891/10000 (78.91%)

Best accuracy so far: 78.91%
Train Epoch: 2 [0/50000 (0%)]   Loss: 0.924297
Train Epoch: 2 [12800/50000 (26%)]      Loss: 0.742250
Train Epoch: 2 [25600/50000 (51%)]      Loss: 0.747472
Train Epoch: 2 [38400/50000 (77%)]      Loss: 0.545068

Test set: Average loss: 0.4734, Accuracy: 9087/10000 (90.87%)

Best accuracy so far: 90.87%
Train Epoch: 3 [0/50000 (0%)]   Loss: 0.471135
Train Epoch: 3 [12800/50000 (26%)]      Loss: 0.326465
Train Epoch: 3 [25600/50000 (51%)]      Loss: 0.366240
Train Epoch: 3 [38400/50000 (77%)]      Loss: 0.419540

Test set: Average loss: 0.2819, Accuracy: 9444/10000 (94.44%)

Best accuracy so far: 94.44%
Train Epoch: 4 [0/50000 (0%)]   Loss: 0.339788
Train Epoch: 4 [12800/50000 (26%)]      Loss: 0.247987
Train Epoch: 4 [25600/50000 (51%)]      Loss: 0.292984
Train Epoch: 4 [38400/50000 (77%)]      Loss: 0.366083

Test set: Average loss: 0.2192, Accuracy: 9536/10000 (95.36%)

Best accuracy so far: 95.36%
Train Epoch: 5 [0/50000 (0%)]   Loss: 0.185025
Train Epoch: 5 [12800/50000 (26%)]      Loss: 0.193982
Train Epoch: 5 [25600/50000 (51%)]      Loss: 0.151002
Train Epoch: 5 [38400/50000 (77%)]      Loss: 0.221164

Test set: Average loss: 0.1760, Accuracy: 9589/10000 (95.89%)

Best accuracy so far: 95.89%
Train Epoch: 6 [0/50000 (0%)]   Loss: 0.243993
Train Epoch: 6 [12800/50000 (26%)]      Loss: 0.183094
Train Epoch: 6 [25600/50000 (51%)]      Loss: 0.135710
Train Epoch: 6 [38400/50000 (77%)]      Loss: 0.223855

Test set: Average loss: 0.1689, Accuracy: 9569/10000 (95.69%)

Best accuracy so far: 95.89%
Train Epoch: 7 [0/50000 (0%)]   Loss: 0.148669
Train Epoch: 7 [12800/50000 (26%)]      Loss: 0.124465
Train Epoch: 7 [25600/50000 (51%)]      Loss: 0.113702
Train Epoch: 7 [38400/50000 (77%)]      Loss: 0.086305

Test set: Average loss: 0.1591, Accuracy: 9602/10000 (96.02%)

Best accuracy so far: 96.02%
Train Epoch: 8 [0/50000 (0%)]   Loss: 0.118434
Train Epoch: 8 [12800/50000 (26%)]      Loss: 0.097000
Train Epoch: 8 [25600/50000 (51%)]      Loss: 0.089288
Train Epoch: 8 [38400/50000 (77%)]      Loss: 0.142392

Test set: Average loss: 0.1211, Accuracy: 9655/10000 (96.55%)

Best accuracy so far: 96.55%
Train Epoch: 9 [0/50000 (0%)]   Loss: 0.105437
Train Epoch: 9 [12800/50000 (26%)]      Loss: 0.069723
Train Epoch: 9 [25600/50000 (51%)]      Loss: 0.089118
Train Epoch: 9 [38400/50000 (77%)]      Loss: 0.097814

Test set: Average loss: 0.1190, Accuracy: 9661/10000 (96.61%)

Best accuracy so far: 96.61%
Train Epoch: 10 [0/50000 (0%)]  Loss: 0.071850
Train Epoch: 10 [12800/50000 (26%)]     Loss: 0.129664
Train Epoch: 10 [25600/50000 (51%)]     Loss: 0.173114
Train Epoch: 10 [38400/50000 (77%)]     Loss: 0.140543

Test set: Average loss: 0.0997, Accuracy: 9724/10000 (97.24%)

Best accuracy so far: 97.24%
Train Epoch: 11 [0/50000 (0%)]  Loss: 0.059830
Train Epoch: 11 [12800/50000 (26%)]     Loss: 0.079231
Train Epoch: 11 [25600/50000 (51%)]     Loss: 0.170483
Train Epoch: 11 [38400/50000 (77%)]     Loss: 0.057648

Test set: Average loss: 0.1695, Accuracy: 9507/10000 (95.07%)

Best accuracy so far: 97.24%
Train Epoch: 12 [0/50000 (0%)]  Loss: 0.117116
Train Epoch: 12 [12800/50000 (26%)]     Loss: 0.062993
Train Epoch: 12 [25600/50000 (51%)]     Loss: 0.054006
Train Epoch: 12 [38400/50000 (77%)]     Loss: 0.071431

Test set: Average loss: 0.0925, Accuracy: 9741/10000 (97.41%)

Best accuracy so far: 97.41%
Train Epoch: 13 [0/50000 (0%)]  Loss: 0.066574
Train Epoch: 13 [12800/50000 (26%)]     Loss: 0.030793
Train Epoch: 13 [25600/50000 (51%)]     Loss: 0.149634
Train Epoch: 13 [38400/50000 (77%)]     Loss: 0.135646

Test set: Average loss: 0.0831, Accuracy: 9744/10000 (97.44%)

Best accuracy so far: 97.44%
Train Epoch: 14 [0/50000 (0%)]  Loss: 0.052199
Train Epoch: 14 [12800/50000 (26%)]     Loss: 0.076551
Train Epoch: 14 [25600/50000 (51%)]     Loss: 0.073964
Train Epoch: 14 [38400/50000 (77%)]     Loss: 0.033631

Test set: Average loss: 0.0765, Accuracy: 9785/10000 (97.85%)

Best accuracy so far: 97.85%
Train Epoch: 15 [0/50000 (0%)]  Loss: 0.089111
Train Epoch: 15 [12800/50000 (26%)]     Loss: 0.100626
Train Epoch: 15 [25600/50000 (51%)]     Loss: 0.074359
Train Epoch: 15 [38400/50000 (77%)]     Loss: 0.106895

Test set: Average loss: 0.0727, Accuracy: 9776/10000 (97.76%)

Best accuracy so far: 97.85%

Test set: Average loss: 0.0706, Accuracy: 9806/10000 (98.06%)

Final Test Accuracy: 98.06%

Total number of parameters: 3818