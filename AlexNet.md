# Modified AlexNet for Cabbage-Weed Dataset

## Overview
This repository contains a modified version of the AlexNet convolutional neural network architecture, tailored specifically for the Cabbage-Weed dataset. By adding extra layers and fine-tuning the network, we have achieved comparable accuracy to the original AlexNet on this dataset.

## Dataset
The Cabbage-Weed dataset is a collection of images depicting various types of cabbage weeds commonly found in agricultural fields. It consists of high-resolution images captured under different environmental conditions and lighting.

## Modifications
### Architecture
We augmented the original AlexNet architecture by adding additional convolutional, pooling, and fully connected layers to enhance its feature extraction capabilities.

### Preprocessing
Before training the network, we applied standard preprocessing techniques such as resizing, normalization, and data augmentation to improve the model's generalization and robustness.

### Training
The modified AlexNet was trained on the Cabbage-Weed dataset using a combination of stochastic gradient descent (SGD) and transfer learning techniques. We initialized the network with pre-trained weights from ImageNet and fine-tuned it on our dataset.

## Results
Our experiments demonstrate that the modified AlexNet achieves similar accuracy levels to the original architecture on the Cabbage-Weed dataset. We evaluated the model using standard performance metrics such as accuracy, precision, recall, and F1-score.

## Usage
### Dependencies
- Python (>=3.6)
- PyTorch (>=1.0)
- NumPy
- Matplotlib
- scikit-learn

### Training
To train the modified AlexNet on your own dataset, follow these steps:
1. Download the Cabbage-Weed dataset or prepare your custom dataset.
2. Preprocess the images by resizing, normalizing, and augmenting them as necessary.
3. Modify the network architecture in the provided script (`train.py`) according to your requirements.
4. Run the training script and monitor the training progress.
5. Evaluate the trained model using appropriate metrics.

### Inference
Once the model is trained, you can use it for inference on new images by:
1. Loading the trained model weights.
2. Preprocessing the input images using the same techniques as during training.
3. Passing the preprocessed images through the network and obtaining predictions.

## Conclusion
In conclusion, this repository presents a modified version of the AlexNet architecture tailored for the Cabbage-Weed dataset. Through careful modifications and training, we have achieved comparable performance to the original AlexNet on this specific task.

## Acknowledgments
We acknowledge the authors of the AlexNet paper and the creators of the Cabbage-Weed dataset for their valuable contributions to the field of computer vision and agricultural research.

## License
This project is licensed under the MIT License
