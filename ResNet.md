# Modified ResNet for Cabbage-Weed Dataset
## Overview
This repository presents a modified version of the ResNet (Residual Neural Network) architecture adapted for the Cabbage-Weed dataset. We have extended the ResNet model by adding extra layers and fine-tuning it to achieve comparable accuracy to the original ResNet on this dataset.

## Dataset
The Cabbage-Weed dataset comprises images depicting various species of cabbage weeds commonly found in agricultural fields. It encompasses diverse environmental conditions, lighting scenarios, and growth stages of the weeds.

## Modifications
### Architecture
We enhanced the ResNet architecture by incorporating additional residual blocks and adjusting the network depth to better capture the intricate features present in the Cabbage-Weed dataset.

### Preprocessing
Prior to training, we conducted preprocessing steps including image resizing, normalization, and augmentation. These steps aimed to enhance the model's ability to generalize and perform well across different conditions.

### Training
Our modified ResNet model was trained on the Cabbage-Weed dataset utilizing techniques such as stochastic gradient descent (SGD) optimization and transfer learning. We initialized the network with weights pretrained on ImageNet and fine-tuned it on our dataset to adapt to the specific characteristics of cabbage weeds.

## Results
Experimental results demonstrate that the modified ResNet achieves similar accuracy levels to the original architecture on the Cabbage-Weed dataset. We evaluated the model's performance using standard metrics such as accuracy, precision, recall, and F1-score.

## Usage
### Dependencies
- Python (>=3.6)
- PyTorch (>=1.0)
- NumPy
- Matplotlib
- scikit-learn

### Training
To train the modified ResNet on your dataset, follow these steps:
1. Download or prepare the Cabbage-Weed dataset.
2. Preprocess the images by resizing, normalizing, and augmenting them as required.
3. Adjust the network architecture in the provided script (`train.py`) according to your specifications.
4. Execute the training script and monitor the training progress.
5. Assess the trained model's performance using appropriate evaluation metrics.

### Inference
After training, you can employ the model for inference on new images by:
1. Loading the trained model weights.
2. Preprocessing the input images using the same techniques applied during training.
3. Passing the preprocessed images through the network to obtain predictions.

## Conclusion
In summary, this repository offers a modified version of the ResNet architecture tailored for the Cabbage-Weed dataset. Through meticulous modifications and training, we have achieved comparable performance to the original ResNet on this specific task.

## Acknowledgments
We express gratitude to the creators of the ResNet architecture and the contributors to the Cabbage-Weed dataset, whose work has significantly advanced the fields of deep learning and agricultural research.

## License
This project is licensed under the MIT License.

