"""
Script used to convert data into sparse matrix format that
can easily be imported into MATLAB.
Use like this
python convertToSparseMatrix.py ../../../../../data/train_triplets.txt 1000 ../../../../../data/eval/year1_test_triplets_hidden.txt 100
"""
import sys
import time

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

userIdToIndex = {} # Key: userid, Value: Row in matrix
songIdToIndex = {} # Key: songid, Value: Column in matrix
userIndex = 0
songIndex = 0
rows = []
columns = []
entries = []
linesRead = 0

# Read in the training set
f = open(inputFile)

for line in f:
  userid, song, songCount = line.strip().split('\t')

  # Fill in matrix
  if song not in songIdToIndex:
    songIdToIndex[song] = songIndex
    songIndex += 1

  if userid not in userIdToIndex:
    userIdToIndex[userid] = userIndex
    userIndex += 1

  rows.append(userIdToIndex[userid])
  columns.append(songIdToIndex[song])
  entries.append(songCount)

  linesRead += 1
  if linesRead >= numTriplets:
    break

f.close()

numUsersInTraining = userIndex

# Read in the data set with half the user history
f = open(inputFileTest)

linesRead = 0

for line in f:
  userid, song, songCount = line.strip().split('\t')

  # Fill in matrix
  if song not in songIdToIndex:
    songIdToIndex[song] = songIndex
    songIndex += 1

  if userid not in userIdToIndex:
    userIdToIndex[userid] = userIndex
    userIndex += 1

  rows.append(userIdToIndex[userid])
  columns.append(songIdToIndex[song])
  entries.append(songCount)

  linesRead += 1
  if linesRead >= numTripletsTest:
    break

numSongs = songIndex
numUsers = userIndex

# Write to a sparse matrix file that can be read with MATLAB
matrix_file = open('UserSongSparseMatrix' + str(numTriplets) + '_' + str(numTripletsTest) + '.txt', 'w')

for i in range(len(entries)):
    matrix_file.write(str(rows[i]+1) + "\t" + str(columns[i]+1) + "\t" + str(entries[i]) + "\n")

matrix_file.write(str(numUsers-1) + "\t" + str(numSongs-1) + "\t" + str(0.000000) + "\n")

matrix_file.close()

print "Done loading %d triplets!" % (numTriplets + numTripletsTest)

end = time.time()
print "Took %s seconds" % (end - start)

print "Number of users", numUsers
print "Number of songs", numSongs
print "You need to predict for the last %s users" % (numUsers - numUsersInTraining)
