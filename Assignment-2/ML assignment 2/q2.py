import numpy
import pandas
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def calculate_accuracy(testOutputs, testLabels):
    # Calculate the accuracy of the model
    correct = 0
    for i in range(len(testLabels)):
        if testOutputs[i] == testLabels[i]:
            correct += 1
    accuracy = correct / len(testLabels)
    return accuracy

def encode_categorial_variable(data):
    # divide the data into features and labels
    labels = data[:, 0]
    features = data[:, 1:]
    # encode the labels
    labels = labels.astype(int)
    return features, labels

def standard_scalar_normalisation(data):
    # Standardise the numpy data (except the first column)
    for columns in range(1, len(data[0])):
        data[:, columns] = (data[:, columns] - numpy.mean(data[:, columns])) / numpy.std(data[:, columns])
    return data

def read_data(filePath):
    dataPandas = pandas.read_csv(filePath, sep=',', header=None) 
    data = dataPandas.to_numpy()
    # data.shape = (178, 14)
    # First column is label and the next 13 are features 
    return data

def random_divide(data):
    # Randomly divide the data into 80% training and 20% testing
    indexes = numpy.random.permutation(len(data))
    trainData = data[indexes[:int(len(data) * 0.8)]]
    testData = data[indexes[int(len(data) * 0.8):]]
    return trainData, testData
    
def main():
    ##### PART 1 #####
    print("################################################ PART 1 #############################################################")
    # Name of the dataset file
    dataFilePath = 'wine.data'
    # Read the dataset 
    data = read_data(dataFilePath)
    # Standardise the data
    print("Performing standard scalar normalisation...")
    data = standard_scalar_normalisation(data)
    # Randomly divide the data into 80% training and 20% testing
    print("Randomly dividing the dataset into 80% training and 20% testing...")
    trainData, testData = random_divide(data)
    # Encode the labels
    print("Encoding the labels...")
    trainFeatures, trainLabels = encode_categorial_variable(trainData)
    testFeatures, testLabels = encode_categorial_variable(testData)
    print("Done!")
    
    ##### PART 2 #####
    print("################################################ PART 2 ############################################################")
    print("Training Linear Support Vector Machine")
    linearSVM = svm.SVC(kernel='linear')
    linearSVM.fit(trainFeatures, trainLabels)
    testOutputLinearSVM = linearSVM.predict(testFeatures)
    accuracyLinearSVM = calculate_accuracy(testOutputLinearSVM, testLabels)
    print("Accuracy of Linear SVM: ", accuracyLinearSVM)
    
    print("Training Quadratic Support Vector Machine")
    quadraticSVM = svm.SVC(kernel='poly', degree=2)
    quadraticSVM.fit(trainFeatures, trainLabels)
    testOutputQuadraticSVM = quadraticSVM.predict(testFeatures)
    accuracyQuadraticSVM = calculate_accuracy(testOutputQuadraticSVM, testLabels)
    print("Accuracy of Quadratic SVM: ", accuracyQuadraticSVM)
    
    print("Training Radial Basis Function Support Vector Machine")
    radialBasisSVM = svm.SVC(kernel='rbf')
    radialBasisSVM.fit(trainFeatures, trainLabels)
    testOutputRadialBasisSVM = radialBasisSVM.predict(testFeatures)
    accuracyRadialBasisSVM = calculate_accuracy(testOutputRadialBasisSVM, testLabels)
    print("Accuracy of Radial Basis SVM: ", accuracyRadialBasisSVM)
    
    print("Done!")
    
    ##### PART 3 #####
    print("################################################ PART 3 #############################################################")
    optimizer = 'sgd'
    learning_rate = 0.001
    batch_size = 32
    print("Training MLP Classifier")
    print("With optimizer: ", optimizer, " learning rate: ", learning_rate, " batch size: ", batch_size)
    
    print("Training MLP Classifier with 1 hidden layer of 16 neurons")
    MLPClassifier_16 = MLPClassifier(hidden_layer_sizes=(16,), solver=optimizer, learning_rate_init=learning_rate, batch_size=batch_size, max_iter=100000)
    MLPClassifier_16.fit(trainFeatures, trainLabels)
    testOutputMLPClassifier_16 = MLPClassifier_16.predict(testFeatures)
    accuracyMLPClassifier_16 = calculate_accuracy(testOutputMLPClassifier_16, testLabels)
    print("Accuracy of MLP Classifier with 1 hidden layer of 16 neurons: ", accuracyMLPClassifier_16)
    
    print("Training MLP Classifier with 2 hidden layer of 256 neurons and 16 neurons")
    MLPClassifier_256_16 = MLPClassifier(hidden_layer_sizes=(256, 16), solver=optimizer, learning_rate_init=learning_rate, batch_size=batch_size, max_iter=100000)
    MLPClassifier_256_16.fit(trainFeatures, trainLabels)
    testOutputMLPClassifier_256_16 = MLPClassifier_256_16.predict(testFeatures)
    accuracyMLPClassifier_256_16 = calculate_accuracy(testOutputMLPClassifier_256_16, testLabels)
    print("Accuracy of MLP Classifier with 2 hidden layer of 256 neurons and 16 neurons: ", accuracyMLPClassifier_256_16)
    
    print("Done!")
    
    ##### PART 4 #####
    print("################################################ PART 4 #############################################################")
    print("Best Model: MLP Classifier with 1 hidden layer of 16 neurons")
    print("Training the above model with different learning rates")
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    accuracies = []
    for lr in learning_rates:
        MLPClassifier_16_lr = MLPClassifier(hidden_layer_sizes=(16,), solver=optimizer, learning_rate_init=lr, batch_size=batch_size, max_iter=10000)
        testOutputMLPClassifier_16_lr = MLPClassifier_16_lr.fit(trainFeatures, trainLabels).predict(testFeatures)
        accuracyMLPClassifier_16_lr = calculate_accuracy(testOutputMLPClassifier_16_lr, testLabels)
        accuracies.append(accuracyMLPClassifier_16_lr)
        print("Accuracy of MLP Classifier with learning rate: ", lr, " is: ", accuracyMLPClassifier_16_lr)
    print("Done!")
    
    # Plot the graph
    plt.plot(learning_rates, accuracies)
    plt.savefig("learning_rate_vs_accuracy.png")
    
    ##### PART 5 #####
    print("################################################ PART 5 #############################################################")
    print("Applying Forward Selection Method to fiond the best set of features")
    # Get the best set of features
    
    
    ##### PART 5 #####
    print("################################################ PART 5 #############################################################")
    print("Applying Ensemble learning (Max voting technique) using SVM with Quadratic, SVM with Radial Basis Function and MLP Classifier with 1 hidden layer of 16 neurons")
    
if __name__ == '__main__':
    main()