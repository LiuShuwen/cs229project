import data
import numpy as np
import scipy.sparse
import sys
import time

## IMPORT DATA

# Analysing command line arguments
if len(sys.argv) < 4:
  print 'Usage:'
  print '  python %s <triplets training file> <number of triplets> <triplets prediction file> <number of triplets>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]
numTriplets = int(sys.argv[2])
inputFileTest = sys.argv[3]
numTripletsTest = int(sys.argv[4])

start = time.time()

print "Loading Data..."

ratings = data.Data(inputFile, numTriplets, inputFileTest, numTripletsTest)

end = time.time()

print "Took %s seconds" % (end - start)

print ratings.getInfo()


## TRAIN A LATENT FACTOR MODEL

# Parameters
k = 3 # number of latent factors
etaQ = .05 # learning rate for Q
etaP = .05 # learning rate for P
lambdaQ = 3 # regularization for Q
lambdaP = 3 # regularization for P

Q = np.random.rand(ratings.numSongs, k)
P = np.random.rand(ratings.numUsers, k)

numMaxIters = 10

print "Training Latent Factor Model..."

def getObjective():
    objective = 0
    for index in range(ratings.numNonZeros):
        user = ratings.rows[index]
        item = ratings.columns[index]
        rating = ratings.entries[index]
        objective += (rating - np.dot(Q[item,:], P[user,:]))**2
    objective += lambdaQ*np.linalg.norm(Q)**2 + lambdaP*np.linalg.norm(P)**2
    return objective


for iteration in range(numMaxIters):
    print "Iteration", iteration

    for index in range(ratings.numNonZeros):
        user = ratings.rows[index]
        item = ratings.columns[index]
        # rating = R[user,item] does not work
        rating = ratings.entries[index]
        epsilon = 2*(rating - np.dot(Q[item,:], P[user,:]))
        Q[item,:] = Q[item,:] + etaQ*(epsilon*P[user,:] - lambdaQ*Q[item,:])
        P[user,:] = P[user,:] + etaP*(epsilon*Q[item,:] - lambdaP*P[user,:])
        # Need a check for convergence

    print "Objective", getObjective()
