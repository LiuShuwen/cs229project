import data
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sys
import time

## IMPORT DATA

# Analysing command line arguments
if len(sys.argv) < 5:
  print 'Usage:'
  print '  python %s <triplets training file> <number of triplets> <triplets visible history file> <triplets hidden history file> <number of triplets>' % sys.argv[0]
  exit()

inputTrainingFile = sys.argv[1]
numTriplets = int(sys.argv[2])
inputFileTest = sys.argv[3]
inputFileHiddenTest = sys.argv[4]
numTripletsTest = int(sys.argv[5])

start = time.time()

print "Loading Data..."

ratings = data.Data(inputTrainingFile, numTriplets, inputFileTest, inputFileHiddenTest, numTripletsTest)

end = time.time()

print "Took %s seconds" % (end - start)

print ratings.getInfo()


## TRAIN A LATENT FACTOR MODEL

# Parameters
k = 10 # number of latent factors
# /!\ need to make the learning rates a function of the number of users/items; make it decay
# /!\ with the number of iterations??
etaX = .005 # learning rate for X
etaY = .005 # learning rate for Y
lambdaX = 0.01 # regularization for X
lambdaY = 0.01 # regularization for Y

# initializing the matrices X and Y using the SVD with k largest singular values
X, s, vt = scipy.sparse.linalg.svds(ratings.R, k)

X = X.transpose()
Y = np.dot(np.diag(s), vt) # need to form diag(s)? seems kind of wasteful

#X = np.random.rand(ratings.numUsers, k) # u x k
#Y = np.random.rand(k, ratings.numSongs) # k x i

numMaxIters = 100

print "Training Latent Factor Model..."

def getObjective():
    objective = 0
    for index in range(ratings.numNonZeros):
        user = ratings.rows[index]
        item = ratings.columns[index]
        rating = ratings.entries[index]
        objective += (rating - np.dot(X[:, user], Y[:, item]))**2
    objective += lambdaX*np.linalg.norm(X)**2 + lambdaY*np.linalg.norm(Y)**2
    return objective

oldObj = getObjective()
tolerance = 0.0005

# alternating least squares to get optimal P and Q

for iteration in range(numMaxIters):
  #if not iteration % 10:
  print "Iteration %d, objective %f" % (iteration, oldObj)
  # updating X first
  ratingIdx = 0
  for user in range(ratings.numUsers):
    oldUser = user
    weighedSum = np.zeros(k)
    V = lambdaX*np.eye(k)
    while ratingIdx < ratings.numNonZeros and ratings.rows[ratingIdx] == oldUser:
      item = ratings.columns[ratingIdx]
      weighedSum += ratings.R[user, item]*Y[:, item]
      #V += np.dot(Y[:,item], Y[:, item].transpose()) # outer product method exists?
      V += np.outer(Y[:,item], Y[:,item])
      ratingIdx += 1
    X[:, user] = scipy.linalg.solve(V, weighedSum)
    
  # now we update Y
  for item in range(ratings.numSongs):
    weighedSum = np.zeros(k)
    W = lambdaY*np.eye(k)
    for userIdx in range(ratings.R.indptr[item], ratings.R.indptr[item + 1]):
      weighedSum += ratings.R[ratings.R.indices[userIdx], item] * X[:, user]
      #W += np.dot(X[:, user], X[:, user].transpose())
      W += np.outer(X[:, user], X[:, user])
    Y[:, item] = scipy.linalg.solve(W, weighedSum)
  
  newObj = getObjective()  
  if abs(newObj - oldObj)/oldObj < tolerance:
    break
  oldObj = newObj

# prediction time baby
scores = np.dot(X[:,ratings.numUsersInTraining:].transpose(), Y)

# how to predict? first, sort the scores for each test user i.e. row by song score
rankings = np.argsort(scores)

correctCount = 0
testIdx = 0
for testUser in range(ratings.numUsersInTraining, ratings.numUsers):
  for song in range(ratings.numUsers):
    if ratings.C_hidden[testUser, rankings[testIdx,-song]] > 0:
      correctCount += 1
    if correctCount > 4:
      break
  testIdx += 1
  
print "We have successfully predicted %d songs" % correctCount