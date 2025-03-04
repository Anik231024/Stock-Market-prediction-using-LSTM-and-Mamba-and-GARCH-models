# Stock price Forecasting with GARCH Models and Neural Networks

This project demonstrates stock price prediction using 4 models and compares the performance of the 4 models. It includes data preprocessing, volatility forecasting, and model training with PyTorch. The first model uses only LSTM for model training. The second model uses only Mamba model for training and evaluation. The third model called Multi Garch LSTM model first forecasts volatility using a family of GARCH models and uses this also as an input along with the dataset as input to the LSTM. The final model called Multi Garch Mamba model first forecasts volatility using a family of GARCH models and uses this also as an input along with the dataset as input to the Mamba model.The dataset have been provided in the file named Stock prediction dataset.
## Overview

The script performs the following tasks:

1. **Data Loading and Preprocessing:**
   - Loads stock market data from a CSV file.
   - Normalizes the data.
   - Evaluation of dataset 
   - Calculates rolling standard deviation to estimate volatility.
   - Applies various GARCH models to forecast volatility.

2. **Data Preparation:**
   - Prepares the data for training and validation.
   - Reshapes and splits the data into training and validation sets.

3. **Model Training:**
   - Defines a neural network model using PyTorch.
   - Trains the model on the preprocessed data.

4. **Evaluation:**
   - Evaluates the model performance.
   - Visualizes the predicted vs. actual values.
   - Calculates performance metrics such as MSE, RMSE, MAE, R^2, and MAPE.

5. **Post-Processing:**
   - Converts predicted values back to the original scale.
   - Computes percentage change of the predicted closing prices.
   - Saves the results to a CSV file.

## Requirements

The project requires the following Python libraries:

- `torch` - PyTorch for deep learning.
- `arch` - ARCH package for GARCH models.
- `numpy` - Numerical operations.
- `pandas` - Data manipulation.
- `scikit-learn` - For metrics and preprocessing.
- `matplotlib` - For plotting.

Install the required libraries using:


pip install -r requirements.txt

The scripts are .ipynb files

Download the files from the repository and unzip them.
Create a google colab account and upload the notebook in google colab and run the models using the dataset. The names of the files for the 4 models are garch_lstm_experiment.ipynb , Garch+Mamba_Experiment.ipynb, lstm_experiment.ipynb , Mamba_Experiment.ipynb 



# Portfolio Optimization Project

## Overview

This project aims to optimize a stock portfolio using the Multi-Objective Optimization (MOO) approach.Portfolio optimization 
was conducted using the NSGA-II Genetic Algorithm, and results showed that optimization 
based on predicted prices yielded higher returns than using historical data alone. Additionally, 
a comparison between equal-weighted portfolios and those optimized for the best Sharpe ratio 
revealed that portfolios based on the Sharpe ratio achieved superior returns.

## Prerequisites

Before running the code, ensure you have the following Python packages installed. You can install them using the provided `requirements.txt` file.

- **numpy**: Numerical operations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **pymoo**: Multi-objective optimization library
- **pandas_datareader**: Access financial data from various sources

Download the files from the repository and unzip them.
Create a google colab account and upload the notebook in google colab and run the models using the dataset 'Predicted prices for portfolio optimisation' for predicted NSG2.ipynb and 'Stock prediction dataset' for
historical price portfolio optimization nsga2.ipynb 