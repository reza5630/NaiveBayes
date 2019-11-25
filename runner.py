# Example of calculating class probabilities
# Wheat Seeds Dataset, where columns shows the values for
# Area.
# Perimeter.
# Compactness
# Length of kernel.
# Width of kernel.
# Asymmetry coefficient.
# Length of kernel groove.
# Class (1, 2, 3)

from csv import reader
from random import seed
from math import sqrt
from math import exp
from math import pi


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


# Test calculating wheat class probabilities
seed(1)
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

# convert class column to integers
# str_column_to_int(dataset, len(dataset[0]) - 1)

# summaries = summarize_by_class(dataset)
#
# probabilities = calculate_class_probabilities(summaries, dataset[197])
# probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
# print(probabilities)

success = 0
for data in dataset:
    sampleData = dataset.copy()
    sampleData.remove(data)
    probabilities = calculate_class_probabilities(summarize_by_class(sampleData), data)
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    if data[7] == probabilities[0][0]:
        success += 1
success = success * 100 / len(dataset)
print("\nsuccess rate " + str(round(success, 2)) + "%")
