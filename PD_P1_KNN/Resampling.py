from KNN.KNN_Classifier import KNN_Classifier
import random
import plotly.graph_objects as go
# Note the resampling is done solely on the K means classifier to obtain optimal value for k

class Resampling:

    def __init__(self, data_in, folds, temp):
        """
        :param folds: the number of partitions to split the data in cross-validation
        :param data_in: list of tuples in form [(x, y, z, PHASE), (...
        :param temp: the temperature of given points
        """
        self.n_data_p = len(data_in)
        self.n_partitions = folds
        self.temp = temp
        self.data_in = data_in

        # A list of all the errors over k, idex is k-1
        self.errors = []

        # The random shuffle of partitions
        self.prtin_list = self.partition(self.n_data_p, self.n_partitions)
        self.KNN = KNN_Classifier()



    def find_optimal_k(self, max_k):
        """
        Main method which invokes cross-validation methods and optimizes for k

        :return: an integer k which is optimal (in terms of error rate)
        """

        for k in range(1, max_k +1):
            self.errors.append(self.get_err_given_K(k))

        return self.errors.index(min(self.errors)) + 1



    def get_err_given_K(self, k):
        """
        Method splits data into partitions, runs the classification on specified k and gets the average error of each partition
        :param k: the value of k which KNN should classify on
        :return: the average error of all partitions when classified at specified k
        """


        prtin_errors = []
        for i in range(self.n_partitions):
            # Find error on each partition
            train = []

            # Holding all points of just (x, y, z) : PHASE
            test = {}
            for prtin, index in zip(self.prtin_list, range(len(self.prtin_list))):
                if prtin == i:
                    x, y, z, ph = self.data_in[index]
                    test[(x, y, z)] = ph
                else:
                    train.append(self.data_in[index])

            # Get the error for this partition
            p_err = self.getError(train, test, k)
            prtin_errors.append(p_err)

        return sum(prtin_errors) / len(prtin_errors)



    def partition(self, l, k):
        """
        Randomly partition the data
        :param l: the number of data points
        :param k: the number of partitions desired
        :return: a list of length l, with only integers from 1 to k, randomly shuffled
        i.e. [1, 1, 3, 2, 3, 5, 1, 1, 2...]
        """
        prtitn_list = []
        for prtitn in range(k):
            prtitn_list += [prtitn for _ in range(l // k)]

        for i in range(l % k):
            prtitn_list.append(i)


        random.shuffle(prtitn_list)
        return prtitn_list



    def getError(self, train, test, k):
        """
        Runs the classifier with given k and determines the error given the train and test data

        Error function: sum([1 for each test point if classifier gets it wrong]) / len of all test points

        :param train: list of tuples for its given data
        :param test: dict (x, y, z) : Phase
        :param k: the value of k to run the classifier
        :return: its error, provided the error function above
        """
        return sum([1 for p in test if self.KNN.classify(p[0], p[1], p[2], train, k) != test[p]]) / len(test)



    def display_ERR_over_K(self):
        """
        Display a plot of the error rate for each k at a specified partition fold

        :return: void, display plotly figure
        """

        # self.errors holds the average error rate of each of its partitions of specified k
        # The index of each value in self.errors is k-1
        fig = go.Figure(data=go.Scatter(x=[k for k in range(1, len(self.errors) + 1)], y= self.errors))
        fig.update_xaxes(title_text='K value')
        fig.update_yaxes(title_text='Mean error rate')
        fig.show()
        return

