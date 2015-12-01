# Latent Factor Model

import numpy as np
import scipy.sparse
import sys
import time


## IMPORT THE DATA

# Need to import set of half playlists too

# Analysing command line arguments
if len(sys.argv) < 3:
  print 'Usage:'
  print '  python %s <triplets file> <number of triplets>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]
numTriplets = int(sys.argv[2])

start = time.time()

userIdToIndex = {} # Key: userid, Value: Row in matrix
songIdToIndex = {} # Key: songid, Value: Column in matrix
userIndex = 0
songIndex = 0
rows = []
columns = []
entries = []
linesRead = 0
f = open(inputFile)

for line in f:
  userid, song, songCount = line.strip().split('\t')

  # Fill in indices
  if song not in songIdToIndex:
    songIdToIndex[song] = songIndex
    songIndex += 1

  if userid not in userIdToIndex:
    userIdToIndex[userid] = userIndex
    userIndex += 1

  # Fill in rows, columns and entries
  rows.append(userIdToIndex[userid])
  columns.append(songIdToIndex[song])
  entries.append(int(songCount))

  linesRead += 1
  if linesRead >= numTriplets:
    break

numSongs = songIndex
numUsers = userIndex
numNonZeros = len(entries)

f.close()

R = scipy.sparse.coo_matrix((entries, (rows, columns)), (numUsers, numSongs), dtype = np.float)

print "Done loading %d triplets!" % numTriplets

end = time.time()

print "Took %s seconds" % (end - start)

print "Number of non-zero entries", numNonZeros
print "Number of songs", numSongs
print "Number of users", numUsers

## TRAIN A LATENT FACTOR MODEL

# Parameters
k = 3 # number of latent factors
etaQ = .05 # learning rate for Q
etaP = .05 # learning rate for P
lambdaQ = 3 # regularization for Q
lambdaP = 3 # regularization for P

Q = np.random.rand(numSongs, k)
P = np.random.rand(numUsers, k)

numMaxIters = 10

print "Training Latent Factor Model..."

def getObjective():
    objective = 0
    for index in range(numNonZeros):
        user = rows[index]
        item = columns[index]
        rating = entries[index]
        objective += (rating - np.dot(Q[item,:], P[user,:]))**2
    objective += lambdaQ*np.linalg.norm(Q)**2 + lambdaP*np.linalg.norm(P)**2
    return objective


for iteration in range(numMaxIters):
    print "Iteration", iteration

    for index in range(numNonZeros):
        user = rows[index]
        item = columns[index]
        # rating = R[user,item] does not work
        rating = entries[index]
        epsilon = 2*(rating - np.dot(Q[item,:], P[user,:]))
        Q[item,:] = Q[item,:] + etaQ*(epsilon*P[user,:] - lambdaQ*Q[item,:])
        P[user,:] = P[user,:] + etaP*(epsilon*Q[item,:] - lambdaP*P[user,:])
        # Need a check for convergence

    print "Objective", getObjective()
