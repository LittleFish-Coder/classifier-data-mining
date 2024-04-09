# Classifier-Data-Mining

To run the program, see the [Usage](#Usage) section at the end of the document.

## Introduction

KNN (K-Nearest Neighbors) is a classification algorithm that classifies a data point based on the majority class of its k nearest neighbors. The algorithm works by calculating the distance between the data point and all other data points in the dataset. It then selects the k nearest neighbors and assigns the data point to the majority class of the k neighbors.

## Data Preprocessing

### fill the missing values

- marital_status: default to be Single(S)
- num_children_at_home: fill with the mean value (round to the nearest integer)
- member_card: default value to be Basic

### normalize the numerical attributes

- age: min-max normalization (value to be between 0 and 1)
- yearly_income: min-max normalization (value to be between 0 and 1)
- num_children_at_home: min-max normalization (value to be between 0 and 1)

### encode the categorical attributes

- marital_status: one-hot encoding
- member_card: map to integer values (Basic: 0, Normal: 1, Silver: 2, Gold: 3)

## KNN Classifier

### fit

The fit method of the KNN classifier takes the training data as input and stores it in the model.

### predict

The predict method of the KNN classifier takes the test data as input and returns the predicted labels for the test data.

### evaluate

The evaluate method of the KNN classifier takes the test data and the true labels as input and returns the accuracy of the model.

## Results

### Text file: result\_{k}.txt

The text file contains the following information:

```
{id1 attribute1, id2 attribute2, ...} member_card = {label}
```

### Comparison

| k   | Accuracy |
| --- | -------- |
| 3   | 0.66     |
| 4   | 0.62     |
| 5   | 0.64     |
| 6   | 0.64     |
| 9   | 0.67     |
| 10  | 0.67     |

## Usage

clone the repository and run the following command in the terminal:

```bash
git clone https://github.com/LittleFish-Coder/classifier-data-mining.git
```

```bash
cd classifier-data-mining
```

```bash
python nm6121030.py
```

add arguments to the command line to change the default values of the program.

- `--k` to specify the number of neighbors for the KNN algorithm. Default value is 3.

### Example

```bash
python nm6121030.py --k 5
```
