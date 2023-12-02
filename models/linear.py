from constants import LINEAR
from evaluation.regression_evaluation import evaluateRegressionModel
from graphing.graphing import graphCorrelationCoefficients, graphLinearRegression

def executeLinearRegression(train_x, train_y, test_x, test_y, verbose):
  """
  Executes Linear Regression and graphs the results.
  """

  # Predict test data
  regression_evaluation_metrics = evaluateRegressionModel(LINEAR, train_x, train_y, test_x, test_y, verbose=verbose)
  graphCorrelationCoefficients(LINEAR, regression_evaluation_metrics.correlation_coefficients, verbose)
  graphLinearRegression(test_x, test_y, 'actual', verbose)
  graphLinearRegression(test_x, regression_evaluation_metrics.predictions, 'predicted', verbose)
