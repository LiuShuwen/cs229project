"""
Script used to convert data into sparse matrix format that
can easily be imported into MATLAB.
Use like this
python convertToSparseMatrix.py ../../../../../data/train_triplets.txt 1000 ../../../../../data/eval/year1_test_triplets_visible.txt ../../../../../data/eval/year1_test_triplets_hidden.txt 100
"""
import sys
import time

# Analysing command line arguments
if len(sys.argv) < 5:
  print 'Usage:'
  print '  python %s <triplets training file> <number of triplets> <triplets visible history file> <triplets hidden history file> <number of triplets>' % sys.argv[0]
  exit()

inputTrainingFile = sys.argv[1]
numTriplets = int(sys.argv[2])
inputTestFile = sys.argv[3]
inputHiddenTestFile = sys.argv[4]
numTripletsTest = int(sys.argv[5])

start = time.time()

userIdToIndex = {} # Key: userid, Value: Row in matrix
songIdToIndex = {} # Key: songid, Value: Column in matrix
userIndex = 0
songIndex = 0
rows = []
columns = []
entries = []
linesRead = 0

maxLines = numTriplets

for inputFile in [inputTrainingFile, inputTestFile, inputHiddenTestFile]:
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
      if linesRead >= maxLines:
        break

    if inputFile == inputTrainingFile:
        numUsersInTraining = userIndex
        maxLines = numTripletsTest

    if inputFile == inputTestFile:
        numSongs = songIndex
        numUsers = userIndex
        numNonZeros = len(entries)
        rows = rows
        columns = columns
        entries = entries

        # Write to a sparse matrix file that can be read with MATLAB
        matrix_file = open('UserSongSparseMatrix' + str(numTriplets) + '_' + str(numTripletsTest) + '.txt', 'w')
        for i in range(len(entries)):
            matrix_file.write(str(rows[i]+1) + "\t" + str(columns[i]+1) + "\t" + str(entries[i]) + "\n")
        #matrix_file.write(str(numUsers-1) + "\t" + str(numSongs-1) + "\t" + str(0.000000) + "\n")
        matrix_file.close()

        # reset everything to zero to read in the hidden matrix
        rows = []
        columns = []
        entries = []

    if inputFile == inputHiddenTestFile:
        # Write to a sparse matrix file that can be read with MATLAB
        matrix_file_test = open('UserSongSparseMatrixTest' + str(numTriplets) + '_' + str(numTripletsTest) + '.txt', 'w')
        for i in range(len(entries)):
            matrix_file_test.write(str(rows[i]+1) + "\t" + str(columns[i]+1) + "\t" + str(entries[i]) + "\n")
        #matrix_file_test.write(str(userIndex-1) + "\t" + str(songIndex-1) + "\t" + str(0.000000) + "\n")
        matrix_file_test.close()

    f.close()

print "Done loading %d triplets!" % (numTriplets + numTripletsTest)

end = time.time()
print "Took %s seconds" % (end - start)

print "Number of users", numUsers
print "Number of songs", numSongs
print "You need to predict for the last %s users" % (numUsers - numUsersInTraining)
