import sys

# Analysing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <triplets file>' % sys.argv[0]
  exit()

inputFile = sys.argv[1]

f = open(inputFile)
songCount = {}
for line in f:
    _, song, _ = line.strip().split('\t')
    if song in songCount:
        songCount[song] += 1
    else:
        songCount[song] = 1
        
f.close()

# re-order songs by popularity
songsByPopularity = sorted(songCount.keys(), key = lambda s: songCount[s], \
                           reverse = True)