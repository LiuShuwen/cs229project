import data
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
import time

## IMPORT DATA

# Analysing command line arguments
if len(sys.argv) < 6:
  print 'Usage:'
  print '  python %s <triplets training file> <number of triplets> <triplets visible history file>' + \
        '<triplets hidden history file> <number of triplets> <number of factors k>' % sys.argv[0]
  exit()

inputTrainingFile = sys.argv[1]
numTriplets = int(sys.argv[2])
inputFileTest = sys.argv[3]
inputFileHiddenTest = sys.argv[4]
numTripletsTest = int(sys.argv[5])
k = int(sys.argv[6])

start = time.time()

print "Loading Data..."

ratings = data.Data(inputTrainingFile, numTriplets, inputFileTest, inputFileHiddenTest, numTripletsTest)

end = time.time()

print "Took %s seconds" % (end - start)

print ratings.getInfo()

## TRAIN A LATENT FACTOR MODEL

# Parameters
#k = 25 # number of latent factors
# /!\ need to make the learning rates a function of the number of users/items; make it decay
# /!\ with the number of iterations??
etaX = .005 # learning rate for X
etaY = .005 # learning rate for Y
lambdaX = 2 # regularization for X
lambdaY = 2 # regularization for Y

# initializing the matrices X and Y using the SVD with k largest singular values
X, s, vt = scipy.sparse.linalg.svds(ratings.R, k)

X = X.transpose()
Y = np.dot(np.diag(s), vt) # need to form diag(s)? seems kind of wasteful

print "Done initializing the factor matrices X and Y"

#X = np.random.rand(k, ratings.numUsers) # u x k
#Y = np.random.rand(k, ratings.numSongs) # k x i

numMaxIters = 10

def getObjective():
    objective = 0
    for index in range(ratings.numNonZeros):
        user = ratings.rows[index]
        item = ratings.columns[index]
        rating = ratings.R[user, item]
        objective += (rating - np.dot(X[:, user], Y[:, item]))**2
    objective += lambdaX*np.linalg.norm(X)**2 + lambdaY*np.linalg.norm(Y)**2
    return objective

oldObj = getObjective()
tolerance = 0.0005

# alternating least squares to get optimal P and Q
start = time.time()
print "Training Latent Factor Model with %d factors and %f, %f as regularization..." % (k, lambdaX, lambdaY)

for iteration in range(numMaxIters):
  print "Iteration %d, objective %f" % (iteration, oldObj)
  ratingIdx = 0
  for user in range(ratings.numUsers):
    oldUser = user
    weighedSum = np.zeros(k)
    V = lambdaX*np.eye(k)
    while ratingIdx < ratings.numNonZeros and ratings.rows[ratingIdx] == oldUser:
      item = ratings.columns[ratingIdx]
      weighedSum += ratings.R[user, item]*Y[:, item]
      V += np.outer(Y[:,item], Y[:,item])
      ratingIdx += 1
    X[:, user] = scipy.linalg.solve(V, weighedSum)
    
  # now we update Y
  for item in range(ratings.numSongs):
    weighedSum = np.zeros(k)
    W = lambdaY*np.eye(k)
    for userIdx in range(ratings.R.indptr[item], ratings.R.indptr[item + 1]):
      weighedSum += ratings.R[ratings.R.indices[userIdx], item] * X[:, user]
      W += np.outer(X[:, user], X[:, user])
    Y[:, item] = scipy.linalg.solve(W, weighedSum)
  
  newObj = getObjective()  
  if abs(newObj - oldObj)/oldObj < tolerance:
    break
  oldObj = newObj

end = time.time()
print "Done performing ALS on X and Y in %d iterations and %f seconds" % (iteration, end - start)

# prediction time baby
print "Predicting unseen songs"
scores = np.dot(X[:,ratings.numUsersInTraining:].transpose(), Y)

## baseline: recommend most popular songs
#scores = ratings.C.sum(0)
#scores = np.asarray(scores)
#scores = np.reshape(scores, scores.shape[1])

# how to predict? first, sort the scores for each test user i.e. row by song score
numPredictions = 500
#ind = np.argpartition(scores, -numPredictions)[-numPredictions:]
#predictions = ind[np.argsort(scores[ind])][::-1]
#rankings = np.argsort(scores)

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

#python -i .\latentFactorModel.py ..\..\data\train_triplets.txt 3000000  ..\..\data\eval\year1_test_triplets_visible.txt ..\..\data\eval\year1_test_triplets_hidden.txt 10000