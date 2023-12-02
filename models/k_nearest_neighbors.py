import numpy as np
import matplotlib.pyplot as plt
from constants import K_NEAREST_NEIGHBORS
from evaluation.regression_evaluation import evaluateRegressionModel
from graphing.graphing import graphCorrelationCoefficients, graphKNearestNeighborsRegression, graphMeanSquaredError

def executeKNearestNeighborsRegression(train_x, train_y, test_x, test_y, k, verbose):
  """
  Executes K-Nearest Neighbors Regression and graphs the results.
  """

  # Predict test data
  regression_evaluation_metrics = evaluateRegressionModel(K_NEAREST_NEIGHBORS, train_x, train_y, test_x, test_y, k, verbose=verbose)
  graphCorrelationCoefficients(K_NEAREST_NEIGHBORS, regression_evaluation_metrics.correlation_coefficients, verbose)
  graphKNearestNeighborsRegression(test_x, test_y, 'actual', k, verbose)
  graphKNearestNeighborsRegression(test_x, regression_evaluation_metrics.predictions, 'predicted', k, verbose)

def determineOptimumKValue(train_x, train_y, test_x, test_y, verbose):
  """
  Determines the optimum K value for K-Nearest Neighbors Regression.
  """

  k_list = np.arange(1, 50, 1)
  knn_dict = {}

  for k in k_list:
    knn_dict[k] = evaluateRegressionModel(K_NEAREST_NEIGHBORS, train_x, train_y, test_x, test_y, k, verbose=verbose).mean_squared_errors

  graphMeanSquaredError(knn_dict, verbose)
