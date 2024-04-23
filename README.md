# Sentiment Analysis using LSTM

This repository contains code for sentiment analysis using LSTM (Long Short-Term Memory) neural networks. The model is trained on the Sentiment140 dataset, which contains 1.6 million tweets labeled with sentiment (0 for negative, 1 for positive).

## Dataset

The Sentiment140 dataset is used for training and testing the model. It contains the following columns:

- **Target**: Sentiment label (0 for negative, 1 for positive)
- **ID's**: Tweet IDs
- **Date**: Date and time of the tweet
- **Flag**: Unknown flag
- **User**: Twitter username
- **Text**: Tweet text

## Preprocessing

The dataset is preprocessed before training the model. The preprocessing steps include:

1. Converting text to lowercase
2. Removing non-alphabetic characters
3. Tokenization
4. Removing stopwords
5. Stemming

## Model Architecture

The model architecture consists of the following layers:

1. Embedding layer: Maps each word to a vector representation
2. Bidirectional LSTM layer: Captures context information from both directions
3. Dense layers: For classification
4. Output layer: Single neuron with sigmoid activation for binary classification

## Training

The model is trained using the Adam optimizer and binary cross-entropy loss function. The training process involves 10 epochs with a batch size of 1024.

## Evaluation

The model is evaluated using various performance metrics:

- Accuracy
- Confusion matrix
- Precision
- Recall
- F1 score
- Area Under the ROC Curve (AUC-ROC)

## Results

The model achieves an accuracy of approximately 75% on the test set. The performance metrics indicate good performance in classifying sentiment from tweets.

## Dependencies

- pandas
- numpy
- nltk
- scikit-learn
- keras
- matplotlib

## Usage

1. Clone the repository:

   ```
   git clone https://github.com/your_username/sentiment-analysis-lstm.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook or Python script to train and evaluate the model.

## Acknowledgments

- [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140)
- [Kaggle API](https://www.kaggle.com/docs/api)

---

Feel free to customize the README according to your preferences and add any additional information you find relevant.


