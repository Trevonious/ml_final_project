import numpy as np
import sklearn.metrics as metrics
from constants import GAUSSIAN_NAIVE_BAYES, K_NEAREST_NEIGHBORS, LINEAR, LOGISTIC
from sklearn import preprocessing
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

class RegressionEvaluationMetrics:
  """
  Class used to store Regression evaluation metrics.
  """

  adjusted_r2_scores: float or np.ndarray
  correlation_coefficients: np.ndarray
  mean_absolute_errors: float or np.ndarray
  mean_squared_errors: float or np.ndarray
  model: object
  predictions: np.ndarray
  r2_scores: float or np.ndarray

  def __init__(self, adjusted_r2_score, correlation_coefficients, mean_absolute_errors, mean_squared_error, model, predictions, r2_score):
    self.adjusted_r2_scores = adjusted_r2_score
    self.correlation_coefficients = correlation_coefficients
    self.mean_absolute_errors = mean_absolute_errors
    self.mean_squared_errors = mean_squared_error
    self.model = model
    self.predictions = predictions
    self.r2_scores = r2_score

def calculateAdjustedR2Score(r2_score, num_observations, num_predictor_variables, verbose):
  """
  Calculates the adjusted R2 score for a given R2 score, number of observations, and number of predictor variables.\n
  This is used to determine if adding more features to a model is beneficial or not.
  """

  return 1 - ((1 - r2_score) * ((num_observations - 1) / (num_observations - num_predictor_variables - 1)))

def calculateCorrelationCoefficient(x, predictions, verbose):
  """
  Calculates the correlation coefficient for each feature in the dataset.
  """

  return np.corrcoef(x, predictions, rowvar=False)


def calculateMeanAbsoluteError(test_y, predictions, verbose):
  """
  Calculates the mean absolute error between the test labels and the predicted labels.
  """

  return metrics.mean_absolute_error(test_y, predictions)


def calculateMeanSquaredError(test_y, predictions, verbose):
  """
  Calculates the mean squared error between the test labels and the predicted labels.
  """

  return metrics.mean_squared_error(test_y, predictions)

def calculateR2Score(test_y, predictions, verbose):
  """
  Calculates the R2 score between the test labels and the predicted labels.
  """

  return metrics.r2_score(test_y, predictions)

def evaluateRegressionModel(model_type: GAUSSIAN_NAIVE_BAYES or K_NEAREST_NEIGHBORS or LINEAR or LOGISTIC, train_x, train_y, test_x, test_y, k=1, verbose=False):
  """
  Evaluates a Regression model using the following metrics:\n
    - Adjusted R2 Scores\n
    - Correlation Coefficients\n
    - Mean Absolute Errors\n
    - Mean Squared Errors\n
    - R2 Scores\n
  """

  model = None
  predictions = []

  print('Predicting labels using ' + model_type + ' Regression...')

  if model_type == GAUSSIAN_NAIVE_BAYES:
    model, predictions = predictLabelsGaussianNaiveBayes(train_x, train_y, test_x, verbose)
  elif model_type == K_NEAREST_NEIGHBORS:
    model, predictions = predictLabelsKNearestNeighbors(train_x, train_y, test_x, k, verbose)
  elif model_type == LINEAR:
    model, predictions = predictLabelsLinear(train_x, train_y, test_x, verbose)
  elif model_type == LOGISTIC:
    model, predictions = predictLabelsLogistic(train_x, train_y, test_x, verbose)

  print('Predictions complete!')

  correlation_coefficients = calculateCorrelationCoefficient(test_x, predictions, verbose)
  mae = calculateMeanAbsoluteError(test_y, predictions, verbose)
  mse = calculateMeanSquaredError(test_y, predictions, verbose)
  r2_score = calculateR2Score(test_y, predictions, verbose)
  adjusted_r2_score = calculateAdjustedR2Score(r2_score, len(test_y), len(test_x[0]), verbose)

  print(str(model_type) + ' Regression evaluation metrics:')
  print('Adjusted R2 Score: ' + str(adjusted_r2_score))
  print('Mean Absolute Error: ' + str(mae))
  print('Mean Squared Error: ' + str(mse))
  print('R2 Score: ' + str(r2_score))
  print()

  return RegressionEvaluationMetrics(adjusted_r2_score, correlation_coefficients, mae, mse, model, predictions, r2_score)

def predictLabelsGaussianNaiveBayes(train_x, train_y, test_x, verbose):
  """
  Predicts the labels for the test data using Gaussian Naive Bayes.
  """

  gnb_regression = GaussianNB()
  gnb_model = gnb_regression.fit(train_x, train_y.ravel())

  return gnb_model, gnb_model.predict(test_x)

def predictLabelsKNearestNeighbors(train_x, train_y, test_x, k, verbose):
  """
  Predicts the labels for the test data using the K-Nearest Neighbors algorithm.
  """

  knn_regression = KNeighborsRegressor(n_neighbors=int(k))
  knn_model = knn_regression.fit(train_x, train_y)

  return knn_model, knn_model.predict(test_x)

def predictLabelsLinear(train_x, train_y, test_x, verbose):
  """
  Predicts the labels for the test data using Linear Regression.
  """

  linear_regression = LinearRegression()
  linear_model = linear_regression.fit(train_x, train_y)

  return linear_model, linear_model.predict(test_x)

def predictLabelsLogistic(train_x, train_y, test_x, verbose):
  """
  Predicts the labels for the test data using Logistic Regression.
  """

  logistic_regression = LogisticRegression()
  logistic_model = logistic_regression.fit(train_x, train_y.ravel())

  return logistic_model, logistic_model.predict(test_x)
