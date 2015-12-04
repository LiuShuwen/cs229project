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
etaQ = .005 # learning rate for Q
etaP = .005 # learning rate for P
lambdaQ = 3 # regularization for Q
lambdaP = 3 # regularization for P

# initializing the matrices P and Q using the SVD with k largest singular values
Q, s, vt = scipy.sparse.linalg.svds(ratings.R, k)

P = np.dot(np.diag(s), vt).transpose()

#Q = np.random.rand(ratings.numSongs, k) # u x k
#P = np.random.rand(ratings.numUsers, k) # i x k

numMaxIters = 1000

print "Training Latent Factor Model..."

def getObjective():
    #print ratings.R.shape, Q.shape, P.transpose().shape
    #objective = np.linalg.norm(ratings.R - np.dot(Q*P.transpose()))**2
    objective = 0
    for index in range(ratings.numNonZeros):
        user = ratings.rows[index]
        item = ratings.columns[index]
        rating = ratings.entries[index]
        objective += (rating - np.dot(Q[user,:], P[item,:]))**2
    objective += lambdaQ*np.linalg.norm(Q)**2 + lambdaP*np.linalg.norm(P)**2
    return objective

oldObj = getObjective()
tolerance = 0.0005

for iteration in range(numMaxIters):
    if not iteration % 50:
      print "Iteration", iteration

    #for user in range(ratings.numUsersInTraining, ratings.numUsers):
    #  for item in range(ratings.numSongs):
    #    rating = ratings.R[user, item]
    #    epsilon = 2*(rating - np.dot(Q[user,:], P[item,:]))
    #    Q[user,:] = Q[user,:] + etaQ*(epsilon*P[item,:] - lambdaQ*Q[user,:])
    #    P[item,:] = P[item,:] + etaP*(epsilon*Q[user,:] - lambdaP*P[item,:])

    for index in range(ratings.numNonZeros):
        user = ratings.rows[index]
        item = ratings.columns[index]
        # rating = R[user,item] does not work
        rating = ratings.entries[index]
        epsilon = 2*(rating - np.dot(Q[user,:], P[item,:]))
        Q[user,:] = Q[user,:] + etaQ*(epsilon*P[item,:] - lambdaQ*Q[user,:])
        P[item,:] = P[item,:] + etaP*(epsilon*Q[user,:] - lambdaP*P[item,:])
        # Need a check for convergence
    newObj = getObjective()
    if abs(newObj - oldObj)/oldObj < tolerance:
      break
    oldObj = newObj

print "Objective", newObj
