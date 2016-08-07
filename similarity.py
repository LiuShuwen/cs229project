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

ratingType = 0

print "Using rating type", ratingType

ratings = data.Data(inputTrainingFile, numTrainingUsers, inputFileTest, inputFileHiddenTest, numTestUsers, ratingType)

end = time.time()

print "Took %s seconds" % (end - start)

print ratings.getInfo()


## COSINE SIMILARITY

# parameters
alphaFlag = True
mapList = []
for alpha in range(1, 10, 1):
    alpha = float(alpha)/10
    print alpha
    if alpha == None:
        alphaFlag = False

    print "Normalizing R..."

    start = time.time()

    # Get diagonal matrix of inverse user norms
    normVec = np.sqrt(ratings.R.multiply(ratings.R).sum(1))

    # Normalize rows of R (with weighting alpha)
    if alphaFlag:
        Rn1 = ratings.R.multiply(scipy.sparse.csc_matrix(1/np.power(normVec, 2*alpha)))
        Rn2 = ratings.R.multiply(scipy.sparse.csc_matrix(1/np.power(normVec, 2*(1 - alpha))))
    else:
        # CHANGED THIS TO L1 NORM
        Rn = normalize(ratings.R, norm='l2', axis=1)

    end = time.time()

    print "Normalized R in %s seconds!" % (end - start)

    start = time.time()

    print "Computing Cosine Similarity Matrix..."

    # Compute Cosine User-User Matrix
    if alphaFlag:
        CosineUser = Rn1 * Rn2.transpose()
    else:
        CosineUser = Rn * Rn.transpose()

    end = time.time()

    print "Computed Cosine Similarity Matrix in %s seconds!" % (end - start)

    start = time.time()

    print "Convert to Binary..."

    # Binary array of the counts
    BinaryCount = ratings.C.astype(bool).astype(float)

    print "Computing Scores..."

    # Scores
    scores = CosineUser[ratings.numUsersInTraining:,:] * BinaryCount
    scores = np.asarray(scores.todense())

    end = time.time()

    print "Computed Scores in %s seconds!" % (end - start)

    ## PREDICTION
    print "Predicting unseen songs"

    #numPredictions = 500


    for numPredictions in range(500,501):
        print "numPredictions", numPredictions
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

        mapList.append(mAP)

        print "Mean Average Precision at %d: %f; computed in %f" % (numPredictions, mAP, end - start)


print mapList
