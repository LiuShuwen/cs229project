import data
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sklearn.preprocessing import normalize
import sys
import time

# python similarity.py ../../../../../data/train_triplets.txt 10000 ../../../../../data/eval/year1_test_triplets_visible.txt ../../../../../data/eval/year1_test_triplets_hidden.txt 1000

## IMPORT DATA

# Analysing command line arguments
if len(sys.argv) < 5:
  print 'Usage:'
  print '  python %s <triplets training file> <number of triplets> <triplets visible history file> <triplets hidden history file> <number of triplets>' % sys.argv[0]
  exit()

inputTrainingFile = sys.argv[1]
numTrainingUsers = int(sys.argv[2])
inputFileTest = sys.argv[3]
inputFileHiddenTest = sys.argv[4]
numTestUsers = int(sys.argv[5])

start = time.time()

print "Loading Data..."

ratings = data.Data(inputTrainingFile, numTrainingUsers, inputFileTest, inputFileHiddenTest, numTestUsers, 3)

end = time.time()

print "Took %s seconds" % (end - start)

print ratings.getInfo()


## COSINE SIMILARITY

# parameters
#alpha = 0.5

print "Normalizing R..."

start = time.time()

# Get diagonal matrix of inverse item norms
#normVec = np.sqrt(ratings.R.multiply(ratings.R).sum(0))

# Normalize columns of R (with weighting alpha)
#Rn1 = ratings.R.multiply(scipy.sparse.csc_matrix(1/np.power(normVec, 2*alpha)))
#Rn2 = ratings.R.multiply(scipy.sparse.csc_matrix(1/np.power(normVec, 2*(1 - alpha))))

Rn = normalize(ratings.R, norm='l2', axis=0)

end = time.time()

print "Normalized R in %s seconds!" % (end - start)

start = time.time()

print "Computing Cosine Similarity Matrix..."

# Compute Cosine Item-Item Matrix
CosineItem = Rn.transpose() * Rn

end = time.time()

print "Computed Cosine Similarity Matrix in %s seconds!" % (end - start)

start = time.time()

print "Convert to Binary..."

# Binary array of the counts
BinaryCount = ratings.C.astype(bool).astype(int)

print "Computing Scores..."

# Scores
scores = BinaryCount[ratings.numUsersInTraining:,:] * CosineItem
scores = np.asarray(scores.todense())

end = time.time()

print "Computed Scores in %s seconds!" % (end - start)

print scores.shape

## PREDICTION
print "Predicting unseen songs"

numPredictions = 500

start = time.time()

mAP = 0.
testIdx = 0
for testUser in range(ratings.numUsersInTraining, ratings.numUsers):
  # sorting scores to get |numPredictions| highest song indices
  ind = np.argpartition(scores[testIdx, :], -numPredictions)[-numPredictions:]
  predictions = ind[np.argsort(scores[testIdx, ind])][::-1]
  mAP += ratings.averagePrecision(testUser, predictions, numPredictions)
  testIdx += 1

end = time.time()

mAP /= (testIdx)

print "Mean Average Precision at %d: %f; computed in %f" % (numPredictions, mAP, end - start)
