import argparse
import glob
import numpy as np
import os
import pandas as pd
from graphing.graphing import graphDataDistributions
from models.gaussian_naive_bayes import executeGaussianNaiveBayesRegression
from models.k_nearest_neighbors import determineOptimumKValue, executeKNearestNeighborsRegression
from models.linear import executeLinearRegression
from models.logistic import executeLogisticRegression
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', help='Verbose output', type=bool, default=False)
parser.add_argument('-k', '--k_value', help='K value to use for KNN', type=int, default=1)
args = parser.parse_args()
k = int(args.k_value)
verbose = args.verbose

##################
# Main Execution #
##################

print()
print('#################################')
print('# Mobile Phone Price Regression #')
print('#################################')
print()
print('Beginning execution...')
print()

# Get current path
path = os.getcwd()
# Get csv files in data folder
csv_files = glob.glob(os.path.join(path + '\\data', "*.csv"))
data_without_labels = []
data_with_labels = []

print('Reading in data...')

# Read in data from csv files
for f in csv_files:
  
  if 'train' in f:
    data_with_labels = np.array(pd.read_csv(f), dtype=np.float32)

    if verbose:
      print('Data with labels shape: ' + str(data_with_labels.shape))
      print('Data with labels columns:')
      print('\tColumn 0: battery_power')
      print('\tColumn 1: blue')
      print('\tColumn 2: clock_speed')
      print('\tColumn 3: fc')
      print('\tColumn 4: four_g')
      print('\tColumn 5: int_memory')
      print('\tColumn 6: m_dep')
      print('\tColumn 7: mobile_wt')
      print('\tColumn 8: n_cores')
      print('\tColumn 9: pc')
      print('\tColumn 10: px_height')
      print('\tColumn 11: px_width')
      print('\tColumn 12: ram')
      print('\tColumn 13: sc_h')
      print('\tColumn 14: sc_w')
      print('\tColumn 15: talk_time')
      print('\tColumn 16: three_g')
      print('\tColumn 17: touch_screen')
      print('\tColumn 18: wifi')
      print('\tColumn 19: price_range')

  # Unused for now. Can uncomment if it's ever needed.
  # elif 'test' in f:
  #   data_without_labels = np.array(pd.read_csv(f), dtype=np.float32)

  #   if verbose:
  #     print('Data without labels shape: ' + str(data_without_labels.shape))
  #     print('Data without labels columns:')
  #     print('\tColumn 0: id')
  #     print('\tColumn 1: battery_power')
  #     print('\tColumn 2: blue')
  #     print('\tColumn 3: clock_speed')
  #     print('\tColumn 4: dual_sim')
  #     print('\tColumn 5: fc')
  #     print('\tColumn 6: four_g')
  #     print('\tColumn 7: int_memory')
  #     print('\tColumn 8: m_dep')
  #     print('\tColumn 9: mobile_wt')
  #     print('\tColumn 10: n_cores')
  #     print('\tColumn 11: pc')
  #     print('\tColumn 12: px_height')
  #     print('\tColumn 13: px_width')
  #     print('\tColumn 14: ram')
  #     print('\tColumn 15: sc_h')
  #     print('\tColumn 16: sc_w')
  #     print('\tColumn 17: talk_time')
  #     print('\tColumn 18: three_g')
  #     print('\tColumn 19: touch_screen')
  #     print('\tColumn 20: wifi')

print('Preparing data...')

# Populate feature data
data_with_labels_x = data_with_labels[:,:20]
# Populate label data
data_with_labels_y = data_with_labels[:,20]
# Split data into training (70%) and test data (30%)
train_x, test_x, train_y, test_y = train_test_split(data_with_labels_x, data_with_labels_y, train_size=0.7)
train_y = np.array([train_y]).transpose()
test_y = np.array([test_y]).transpose()

if verbose:
  print('Training data shape: ' + str(train_x.shape))
  print('Training labels shape: ' + str(train_y.shape))
  print('Test data shape: ' + str(test_x.shape))
  print('Test labels shape: ' + str(test_y.shape))

print()

# Graph data distributions
graphDataDistributions(data_with_labels_x, verbose)

# Scale data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

if k > 1:
  executeGaussianNaiveBayesRegression(train_x, train_y, test_x, test_y, verbose)

  print()

  executeKNearestNeighborsRegression(train_x, train_y, test_x, test_y, k, verbose)

  print()

  executeLinearRegression(train_x, train_y, test_x, test_y, verbose)

  print()

  executeLogisticRegression(train_x, train_y, test_x, test_y, verbose)
else:

  print('Determine the optimum k value by analyzing the following graph and finding the "elbow"...')
  print('The "elbow" is the point where the graph starts to flatten out...')
  print('Then re-run this script with the optimum k value using the "-k" flag followed by the value...')
  print('See the README for more information...')

  determineOptimumKValue(train_x, train_y, test_x, test_y, verbose)

print()
print('Execution complete!')
print()
