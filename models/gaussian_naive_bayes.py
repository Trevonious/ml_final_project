from constants import GAUSSIAN_NAIVE_BAYES
from evaluation.regression_evaluation import evaluateRegressionModel
from graphing.graphing import graphCorrelationCoefficients, graphGaussianNaiveBayesRegression

def executeGaussianNaiveBayesRegression(train_x, train_y, test_x, test_y, verbose):
  """
  Executes Gaussian Naive Bayes Regression and graphs the results.
  """

  # Predict test data
  regression_evaluation_metrics = evaluateRegressionModel(GAUSSIAN_NAIVE_BAYES, train_x, train_y, test_x, test_y, verbose=verbose)
  graphCorrelationCoefficients(GAUSSIAN_NAIVE_BAYES, regression_evaluation_metrics.correlation_coefficients, verbose)
  graphGaussianNaiveBayesRegression(test_x, test_y, 'actual', verbose)
  graphGaussianNaiveBayesRegression(test_x, regression_evaluation_metrics.predictions, 'predicted', verbose)
  