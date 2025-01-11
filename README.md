# Breast Cancer Classification with PyTorch

This project demonstrates how to build and train a simple neural network using PyTorch to classify breast cancer tumors as malignant or benign.

## Dataset

The project utilizes the Breast Cancer Wisconsin (Diagnostic) Dataset, which is readily available through scikit-learn's `load_breast_cancer` function. This dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast mass.

## Model

A simple feedforward neural network with one hidden layer is employed for classification. It consists of:

- An input layer with 30 features (corresponding to the dataset's features).
- A hidden layer with 64 neurons and ReLU activation.
- An output layer with a single neuron and sigmoid activation for binary classification.

## Dependencies

- Python 3.x
- PyTorch 1.13.1+cu117
- scikit-learn 1.2.2
- torch.nn
- torch.optim

Install these libraries using `pip`:

## Usage

1. **Load and Preprocess Data:** The dataset is loaded using `load_breast_cancer`, split into training and testing sets, and standardized using `StandardScaler`.
2. **Define Model:** The neural network architecture is defined using PyTorch's `nn.Module`.
3. **Train Model:** The model is trained using the Adam optimizer and binary cross-entropy loss.
4. **Evaluate Model:** The trained model is evaluated on both training and testing sets to assess its performance.

## Results

The model achieves high accuracy on both training and testing data, indicating its effectiveness in classifying breast cancer tumors. To see the results, run the code.

## Contributing

Feel free to contribute to this project by suggesting improvements or adding new features.
