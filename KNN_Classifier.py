class KNN_Classifier:

    def __init__(self):
        pass



    def inRange(self, p1, p2, range):
        # Returns whether the points are in range defined by epsilon
        x, y, z, phase2 = p2
        x_, y_, z_ = p1
        return (x_ - range < x < x_ + range) and (y_ - range < y < y_ + range) and (z_ - range < z < z_ + range)



    def dist(self, p1, p2):
        # Returns euclidean distance between two points.
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2



    def resolveTie(self, p, k_nearest, phases):
        """
        In the case of a tie in the k nearest neighbours, base judgement on distance.
        Create a function of weights given distance. Sum(1/d)
        :param p: point trying to classify
        :param k_nearest: all k nearest points found
        :return: a phase based on the distance weights
        """

        # phase_dist := list of tuples (phase, summed 1 / d of each point)
        phase_dist = [(phase, sum([1 / self.dist(p, p_) for p_ in k_nearest if p_[3] == phase])) for phase in phases]
        sorted_dist = sorted(phase_dist, key=lambda x:x[1])[::-1]

        return sorted_dist[0][0]



    def classify(self, x, y, z, train_data, k, phases):
        """
        Will classify a point based on given training observations
        :param x, y, z: The point to predict its phase
        :param train_data: a list of tuples in form: (x, y, z, PHASE)
        :param k: the number of nearest points to use in the prediction
        :param phases: tuple containing all phases with arbitrary data type
        :return: the phase in which the KNN algorithm predicts
        """

        # We are not approaching the exhaustive method of going through every point and computing distance.

        # Alternative faster approach:
        # 1, Iteratively define a range around given point.
        # 2, Get the number of points in range.
        # if number of points equals k, we are done.
        # 3, change range based on the difference of k
        # 4, Go to 2 (is a binary search for size of range)

        # First find the nearest k points

        # Initialize range parameter
        range = 0.2
        min_range, max_range = (0.01, 0.95)

        neighbours_found = 0
        converged = False
        nearest = []
        last_range = 0
        while not converged:

            nearest = []
            nearest = [p for p in train_data if self.inRange((x, y, z), p, range)]

            neighbours_found = len(nearest)

            # Allow it to converge if the change in range from last iteration is negligible
            if neighbours_found == k or abs(range - last_range) < 0.008:
                converged = True

            last_range = range

            if neighbours_found < k:
                # We need to expand range
                min_range = range
                range += (max_range - range) / 2
            elif neighbours_found > k:
                # We need to shorten range
                max_range = range
                range -= (range - min_range) / 2


        # Then classify based on the phases of k points
        # List of tuples := (phase , number of k nearest neighbours in the phase (int))
        phase_list = [(phase , sum([1 for p in nearest if phase == p[3]])) for phase in phases]
        sorted_phases = sorted(phase_list, key=lambda x:x[1])[::-1]

        if sorted_phases[0][1] == sorted_phases[1][1]:
            return self.resolveTie((x, y, z), nearest, phases)
        return sorted_phases[0][0]



    def KNN(self, k, train_data, phases, delta):
        """
        Run the classifier on given data and generate entire ternary diagram

        :param k: the choice of the number of nearest neighbours to look at
        :param train_data:
        :param phases: tuple containing all phases with arbitrary data type
        :param delta: float increment of point for each dimension. Smaller delta implies denser diagram (more points)
        :param delta: [0.01, 1)
        :return: list of tuples: [(x, y, z, PHASE), ()...] for every single point within its density
        """

        # The master list holding all points and their phases
        ternary = []
        ternary += train_data
        # Points in train WITHOUT phase label
        p_train = [(i[0], i[1], i[2]) for i in train_data]

        for x in range(0, 100, int(delta*100)):
            for y in range(0, 100 - x, int(delta*100)):
                x_ = x /100
                y_ = y /100
                z_ = (100 - x - y) / 100

                if (x_, y_, z_) not in p_train:
                    ternary.append((x_, y_, z_, self.classify(x_, y_, z_, train_data, k, phases)))

        return ternary