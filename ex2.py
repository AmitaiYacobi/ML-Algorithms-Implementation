import numpy as np
import sys
import os

train, targets, test, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
train = np.loadtxt(train, delimiter=",")
test = np.loadtxt(test, delimiter=",")
train_targets = np.loadtxt(targets)
output_file = open(output_file, "w")


def minmax_normalization(data_set):
    min_vector = np.min(data_set, axis=0)
    max_vector = np.max(data_set, axis=0)
    for row in data_set:
        for i in range(len(row)):
            row[i] = (row[i] - min_vector[i]) / (max_vector[i] - min_vector[i])
    return data_set

def zscore_normalization(data_set):
    return (data_set - data_set.mean(0)) / data_set.std(0)

def accuracy(predictions, targets):
    counter = 0
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            counter += 1
    return (counter / len(targets)) * 100

class KNN:
    def __init__(self, train_set, targets, k):
        self.train_set = train_set
        self.targets = targets
        self.k = k

    def predict(self, test_row):
        classes = [0, 0, 0]
        distances = [np.linalg.norm(test_row - example) for example in self.train_set]
        sorted_distances = sorted(distances)
        k_nearest_distances = sorted_distances[: self.k]
        k_nearest_neighbors = [self.train_set[distances.index(distance)] for distance in k_nearest_distances]
        for neighbor in k_nearest_neighbors: 
            neighbor_index = np.where(np.all(self.train_set == neighbor, axis=1))[0][0]
            classes[int(self.targets[neighbor_index])] += 1
        return (classes.index(max(classes)))

class Perceptron:
    def __init__(self, train_set, targets):
        self.train_set = train_set
        self.targets = targets
        self.learning_rate = 0.1
    
    def weights_init(self):
        input_size = len(self.train_set[0])
        output_size = 3
        weights = np.zeros((output_size, input_size))
        return weights

    def train(self, num_of_epochs):
        bias_column = np.full(240,1).reshape(240,1)
        self.train_set = np.append(self.train_set, bias_column, axis=1)
        w = self.weights_init()
        for e in range(num_of_epochs):
            if e % 10 == 0 : self.learning_rate *= 0.1
            union_data = list(zip(self.train_set, self.targets))
            np.random.shuffle(union_data)
            for x, y in union_data:
                y_hat = np.argmax(np.dot(w, x))
                y = int(y)
                y_hat = int(y_hat)
                if y != y_hat:
                    w[y, :] = w[y, :] + (self.learning_rate * x)
                    w[y_hat, :] = w[y_hat, :] - (self.learning_rate * x)
        return w
    
    def predict(self, weights, test_row):
        test_row = np.append(test_row, 1)
        return np.argmax(np.dot(weights, test_row))




if __name__ == "__main__":
    train = zscore_normalization(train)
    test = zscore_normalization(test)

    knn = KNN(train, train_targets, 15)
    perceptron = Perceptron(train, train_targets)

    per_weights = perceptron.train(20)
    for row in test:
        knn_yhat = knn.predict(row)
        perceptron_yhat = perceptron.predict(per_weights, row)
        output_file.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {0}, pa: {0}\n")

    
    
    
    
    
    # validation = np.loadtxt("validation_x.txt", delimiter=",")
    # validation_targets = np.loadtxt("validation_y.txt", delimiter=",")
    # validation = minmax_normalization(validation)
    # predictions = []
    # for row in validation:
    #     predict = knn.predict(row)
    #     predictions.append(predict)
    # # print(f"accuracy is: {accuracy(predictions, validation_targets)}%")