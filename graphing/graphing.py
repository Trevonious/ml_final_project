import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
from constants import COLUMNS_WITH_PRICE_RANGE, COLUMNS_WITHOUT_PRICE_RANGE 

def graphCorrelationCoefficients(model_type, correlation_coefficients, verbose):
  """
  Graphs the correlation coefficients for each feature in the dataset.
  """

  print('Graphing ' + model_type + ' Correlation Coefficients...')

  plt.title(model_type + ' Correlation Coefficients')
  plt.rcParams["figure.figsize"] = [10, 10]
  df = pd.DataFrame(correlation_coefficients, columns=COLUMNS_WITH_PRICE_RANGE, index=COLUMNS_WITH_PRICE_RANGE)
  # Hides upper half of correlation matrix
  mask = np.triu(df)
  sns.heatmap(df, annot=True, cbar=False, cmap="plasma", mask=mask)
  plt.show()

def graphDataDistributions(x, verbose):
  """
  Graphs the data distributions for each feature in the dataset.
  """

  print('Graphing data distributions...')
  print()

  df = pd.DataFrame(x, columns=COLUMNS_WITHOUT_PRICE_RANGE)
  fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(10, 10))
  column_index = 0
  row_index = 0

  for column in df:
    sns.histplot(df[column], ax=axs[row_index][column_index], kde_kws=dict(color='plum', edgecolor="k", linewidth=1))

    if (column_index == 3):
      column_index = 0
      row_index += 1
    else:
      column_index += 1

  plt.show()

def graphGaussianNaiveBayesRegression(test_x, test_y, label_type, verbose):
  """
  Graphs the Gaussian Naive Bayes Regression model.
  """

  print('Graphing Gaussian Naive Bayes Regression with ' + label_type + ' labels...')

  df = pd.DataFrame(test_x, columns=COLUMNS_WITHOUT_PRICE_RANGE)
  plt.rcParams["figure.figsize"] = [10, 10]
  plt.rcParams["figure.autolayout"] = True
  fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(15, 15))
  column_index = 0
  row_index = 0

  for column in df:
    sns.regplot(x=df[column], y=test_y, data=df, ax=axs[row_index][column_index], label=COLUMNS_WITHOUT_PRICE_RANGE[row_index * 4 + column_index], color='blue', line_kws={'color': 'red'})
    plt.ylabel('Price Range')
    plt.xlabel('')
    plt.tight_layout()

    if (column_index == 3):
      column_index = 0
      row_index += 1
    else:
      column_index += 1
  
  plt.show()

def graphKNearestNeighborsRegression(test_x, test_y, label_type, k, verbose):
  """
  Graphs the K-Nearest Neighbors Regression model.
  """

  print('Graphing K-Nearest Neighbors Regression with ' + label_type + ' labels...')

  df = pd.DataFrame(test_x, columns=COLUMNS_WITHOUT_PRICE_RANGE)
  plt.rcParams["figure.figsize"] = [10, 10]
  plt.rcParams["figure.autolayout"] = True
  fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(15, 15))
  column_index = 0
  row_index = 0

  for column in df:
    sns.regplot(
      x=df[column],
      y=test_y,
      data=df,
      ax=axs[row_index][column_index],
      label=COLUMNS_WITHOUT_PRICE_RANGE[row_index * 4 + column_index],
      color='blue',
      line_kws={'color': 'red'}
    )
    plt.ylabel('Price Range')
    plt.xlabel('')
    plt.tight_layout()

    if (column_index == 3):
      column_index = 0
      row_index += 1
    else:
      column_index += 1

  plt.show()

def graphLinearRegression(test_x, test_y, label_type, verbose):
  """
  Graphs the Linear Regression model.
  """

  print('Graphing Linear Regression with ' + label_type + ' labels...')

  df = pd.DataFrame(test_x, columns=COLUMNS_WITHOUT_PRICE_RANGE)
  plt.rcParams["figure.figsize"] = [10, 10]
  plt.rcParams["figure.autolayout"] = True
  fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(15, 15))
  column_index = 0
  row_index = 0

  for column in df:
    sns.regplot(x=df[column], y=test_y, data=df, ax=axs[row_index][column_index], label=COLUMNS_WITHOUT_PRICE_RANGE[row_index * 4 + column_index], color='blue', line_kws={'color': 'red'})
    plt.ylabel('Price Range')
    plt.xlabel('')
    plt.tight_layout()

    if (column_index == 3):
      column_index = 0
      row_index += 1
    else:
      column_index += 1
  
  plt.show()

def graphLogisticRegression(test_x, test_y, label_type, verbose):
  """
  Graphs the Logistic Regression model.
  """

  print('Graphing Logistic Regression with ' + label_type + ' labels...')

  df = pd.DataFrame(test_x, columns=COLUMNS_WITHOUT_PRICE_RANGE)
  plt.rcParams["figure.figsize"] = [10, 10]
  plt.rcParams["figure.autolayout"] = True
  fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(15, 15))
  column_index = 0
  row_index = 0

  for column in df:
    sns.regplot(x=df[column], y=test_y, data=df, ax=axs[row_index][column_index], label=COLUMNS_WITHOUT_PRICE_RANGE[row_index * 4 + column_index], logistic=True, color='blue', line_kws={'color': 'red'})
    plt.ylabel('Price Range')
    plt.xlabel('')
    plt.tight_layout()

    if (column_index == 3):
      column_index = 0
      row_index += 1
    else:
      column_index += 1

  plt.show()
  
def graphMeanSquaredError(knn_dict, verbose):
  """
  Graphs the mean squared error for each value of k.
  """

  print('Graphing mean squared error...')
  
  fig, ax = plt.subplots(figsize=(10, 10))
  ax.plot(knn_dict.keys(), knn_dict.values())
  ax.set_xlabel('K Value', fontsize=24)
  ax.set_ylabel('Mean Squared Error', fontsize=24)
  ax.set_title('Plot to Determine Elbow', fontsize=32)
  plt.show()
