from sklearn import svm
import random

class ResampleSVM:

    def __init__(self, data_in, gamma_range=(0.02, 1.5, 0.01), C_range=(200, 25000, 200), k_fold=10):
        # gamma and C are parameters to optimize for the SVM
        self.gamma_range = gamma_range
        self.C_range = C_range
        self.data_in = data_in
        self.k_fold = k_fold
        self.min_g = 0
        self.min_C = 0


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


    def classify_set(self, train, test, g, C):
        """
        Return the error given params

        Implement the SVM to classify all points in test
        :param train: data to fit the SVM [(x, y, z, PHASE), ...]
        :param test: data to test the SVM
        :param g, C: the SVM parameters
        :return: the error rate!

        Error rate defined as sum([1 for all test points if SVM classifies wrong]) / len(test points)

        """
        unlabelled_test = [x[:3] for x in test]

        classifier = svm.SVC(kernel='rbf', gamma=g, C=C)
        classifier.fit(list(map(lambda x: [x[0], x[1], x[2]], train)), list(map(lambda x: x[-1], train)))
        return sum([1 for i, phase in zip(list(range(len(test))), classifier.predict(unlabelled_test)) if test[i][-1] != int(phase)]) / len(test)


    def generate_all_phase(self, train, g, C, delta):
        """
        Get the SVM to classify all points given parameters and training set.

        :return: list of tuples of all points [(x, y, z, PHASE)...]
        """

        points_list = []

        classifier = svm.SVC(kernel='rbf', gamma=g, C=C)
        classifier.fit(list(map(lambda x: (x[0], x[1], x[2]), train)), list(map(lambda x: x[-1], train)))

        for x in range(0, 1000, int(delta*1000)):
            for y in range(0, 1000 - x, int(delta*1000)):
                x_ = x /1000
                y_ = y /1000
                z_ = (1000 - x - y) / 1000
                points_list.append((x_, y_, z_, int(classifier.predict([[x_, y_, z_]])[0])))

        return points_list


    def get_avg_kFold_partitn(self, g, C):
        """
        Return the average error of all k partitions

        :param g:
        :param C:
        :return: float
        """
        len_data = len(self.data_in)
        # First partition the data into k distinct non-overlapping sets
        partitn = self.partition(len_data, self.k_fold)
        # For each partition, use rest for train, get error on the kth partition as test data
        err_total = 0

        for k_split in range(self.k_fold):
            train = []
            test = []
            for i in range(len_data):
                if partitn[i] == k_split:
                    test.append(self.data_in[i])
                else:
                    train.append(self.data_in[i])
            err_total += self.classify_set(train, test, g, C)

        return g, C, err_total / self.k_fold


    def display_error_over_GandC(self):
        """
        Show a 3D surface plot of the error given parameters gamma and C
        :return: nothing, display a plotly figure.
        """
        x = []
        y = []
        z = []
        _2d = []
        g = self.gamma_range[0]
        C = self.C_range[0]
        while g < self.gamma_range[1]:
            l = []
            while C < self.C_range[1]:
                x.append(g)
                y.append(C)
                v = self.get_avg_kFold_partitn(g, C)
                z.append(v)
                l.append(v)
                C += self.C_range[2]
            g += self.gamma_range[2]
            C = self.C_range[0]
            _2d.append(l)

        min_err = min(z)
        self.min_C = y[z.index(min_err)]
        self.min_g = x[z.index(min_err)]

        """
        fig = go.Figure(data=[go.Mesh3d(z=z, x=x, y=y, colorbar_title='Mean Error Rate',
        colorscale=[[0, 'gold'],
                    [0.5, 'mediumturquoise'],
                    [1, 'magenta']],intensity = z,showscale=True)])

        fig.update_layout(scene=dict(
            xaxis_title='GAMMA',
            yaxis_title='COST',
            zaxis_title='MEAN ERROR RATE'))

        fig.show()
        """
        return


