import sys
import time
from recommender import *

# Analysing command line arguments
if len(sys.argv) < 3:
  print 'Usage:'
  print '  python %s <triplets file> <number of triplets>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]
numTriplets = int(sys.argv[2])

start = time.time()
# read data
r = Recommender(inputFile, numTriplets)

end = time.time()
print "Took %s seconds" % (end - start)

print "Number of songs %s" % r.numSongs
print "Number of users %s" % r.numUsers
print r.userSongMatrix
