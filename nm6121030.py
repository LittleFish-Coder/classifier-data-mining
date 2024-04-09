import pandas as pd
import numpy as np
import argparse


def load_data(file_path):
    # Open the file
    with open(file_path, "r") as file:
        # Read the data
        lines = file.readlines()

    data = []
    default_values = {
        "0": None,
        "1": None,
        "2": "Basic",
        "3": None,
        "4": None,
    }
    for line in lines:
        curr_values = default_values.copy()

        # remove the first and last curly braces
        line = line.strip()[1:-1]

        # split the line by commas
        items = line.split(",")

        for item in items:
            id, attribute = item.split()
            curr_values[id] = attribute

        data.append(curr_values)

    return data


def data_preprocessing(data: list):
    # Create the DataFrame
    df = pd.DataFrame(data)

    df = df.rename(
        columns={
            "0": "marital_status",
            "1": "num_children_at_home",
            "2": "member_card",
            "3": "age",
            "4": "year_income",
        }
    )

    # Convert the columns to the correct data types
    df["num_children_at_home"] = df["num_children_at_home"].astype(float)
    df["age"] = df["age"].astype(float)
    df["year_income"] = df["year_income"].astype(float)

    # fill the missing values  -> only for the columns that have missing values
    ## marital_status default to be S
    df["marital_status"] = df["marital_status"].fillna("S")
    ## num_children_at_home default to be mean
    df["num_children_at_home"] = df["num_children_at_home"].fillna(
        int(df["num_children_at_home"].mean())
    )

    # data transformation
    ## one-hot encoding for marital_status
    df = pd.get_dummies(df, columns=["marital_status"])
    ## normalization for age and year_income (min-max normalization)
    df["age"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
    df["year_income"] = (df["year_income"] - df["year_income"].min()) / (
        df["year_income"].max() - df["year_income"].min()
    )
    ## categorical encoding for member_card
    df["member_card"] = df["member_card"].map({"Basic": 0, "Normal": 1, "Silver": 2, "Gold": 3})

    return df


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, dataframe):

        self.X_train = dataframe.drop(["member_card"], axis=1)
        self.y_train = dataframe["member_card"].values

    def predict(self, dataframe):
        self.X_test = dataframe.drop(["member_card"], axis=1)
        self.y_test = dataframe["member_card"].values

        self.y_predictions = []

        #  Calculate the distance between the test data and the training data
        for X in self.X_test.values:
            distances = []
            for X_train in self.X_train.values:
                distance = np.sqrt(np.sum((X - X_train) ** 2))
                distances.append(distance)

            # Get the indices of the k-nearest neighbors
            indices = np.argsort(distances)[: self.k]

            # Get the labels of the k-nearest neighbors
            labels = self.y_train[indices]

            # Get the most common label
            prediction = np.bincount(labels).argmax()

            self.y_predictions.append(prediction)

        return self.y_predictions

    def evaluate(self):
        # Calculate the accuracy
        accuracy = np.mean(self.y_predictions == self.y_test)

        print(f"KNN (k={self.k})")
        print(f"Accuracy: {accuracy:.2f}")


def output_data(input_file, labels, output_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    with open(output_file, "w") as file:
        for i, line in enumerate(lines):
            file.write(line.strip() + f" member_card = {labels[i]}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="number of neighbors")
    args = parser.parse_args()
    k = args.k

    data = load_data("training.txt")
    df = data_preprocessing(data)

    test_data = load_data("test.txt")
    test_df = data_preprocessing(test_data)

    # model classification
    knn = KNN(k=k)
    knn.fit(df)  # training
    prediction = knn.predict(test_df)  # predict
    # print(prediction)
    knn.evaluate()

    # mapping the member_card to the original value
    mapping = {0: "Basic", 1: "Normal", 2: "Silver", 3: "Gold"}
    prediction = [mapping[p] for p in prediction]

    output_data("test.txt", prediction, f"result_{k}.txt")
