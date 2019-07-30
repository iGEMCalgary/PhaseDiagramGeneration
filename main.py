from KNN_Classifier import KNN_Classifier
from Resampling import Resampling
from DisplayTernaryPhaseDiag import Display
import random

ALL_DATA_PATH = 'limestone T2 Gen.csv'

display = Display()
KNN = KNN_Classifier()
# RUN FUNCTIONS
data_read = display.FormatDataFromCSV(ALL_DATA_PATH)
print(data_read)
data_copy = data_read[1].copy()

# Partition the set into training and test randomly
n_train = 60
train = []

for ranIndex in [random.randint(0, len(data_read[1]) - i) for i in range(n_train)]:
    train.append(data_read[1].pop(ranIndex))

test = data_read[1]


display.DisplayTernaryPhaseScatter(train, 5)

display.DisplayTernaryPhaseScatter(KNN.KNN(5, train, display.phases, 0.01), 5)

