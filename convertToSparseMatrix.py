# Convert To Sparse Matrix
import sys
import time

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
matrix_file = open('UserSongSparseMatrix' + str(numTriplets) + '.txt', 'w')

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

numSongs = songIndex
numUsers = userIndex

for i in range(len(entries)):
    matrix_file.write(str(rows[i]+1) + "\t" + str(columns[i]+1) + "\t" + str(entries[i]) + "\n")

matrix_file.write(str(numUsers-1) + "\t" + str(numSongs-1) + "\t" + str(0.000000) + "\n")

matrix_file.close()
f.close()

print "Done loading %d triplets!" % numTriplets

end = time.time()
print "Took %s seconds" % (end - start)
