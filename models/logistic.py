import numpy as np
from constants import LOGISTIC
from evaluation.regression_evaluation import evaluateRegressionModel
from graphing.graphing import graphCorrelationCoefficients, graphLogisticRegression
from sklearn import preprocessing

def executeLogisticRegression(train_x, train_y, test_x, test_y, verbose):
  """
  Executes Logistic Regression and graphs the results.
  """

  # With more time, we wanted to get Logistic Regression working, to classify (binary) whether or not
  # a phone was in the "high" (price_range=2) or "very high" (price_range=3) price range (True=price_range[2,3];False=price_range[0,1]).
  # Attempted to encode to get rid of 0s that are causing exceptions. Still needs work.
  # lab = preprocessing.LabelEncoder()
  # lab.fit(train_y.ravel().tolist())
  # transformed_train_y = lab.transform(train_y.ravel().tolist())
  # transformed_test_y = lab.transform(test_y.ravel().tolist())
  # transformed_train_y = np.array([transformed_train_y]).transpose()
  # transformed_test_y = np.array([transformed_test_y]).transpose()
  # regression_evaluation_metrics = evaluateRegressionModel(LOGISTIC, train_x, transformed_train_y, test_x, transformed_test_y, verbose=verbose)
  # graphCorrelationCoefficients(LOGISTIC, regression_evaluation_metrics.correlation_coefficients, verbose)
  # graphLogisticRegression(test_x, transformed_test_y, verbose)

  # Predict test data
  regression_evaluation_metrics = evaluateRegressionModel(LOGISTIC, train_x, train_y, test_x, test_y, verbose=verbose)
  graphCorrelationCoefficients(LOGISTIC, regression_evaluation_metrics.correlation_coefficients, verbose)

  # Didn't get this working because of the issue mentioned above.
  #graphLogisticRegression(test_x, test_y, 'actual', verbose)
  #graphLogisticRegression(test_x, regression_evaluation_metrics.predictions, 'predicted', verbose)
