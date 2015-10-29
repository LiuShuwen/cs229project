# imports

class Recommender():
  def __init__(self, filename, numTriplets = 1000):
    self.filename = filename
    self.numTriplets = numTriplets
    self.songCount = {} # map song to number of times it's been listened to
    self.userHistory = {}
    self.readTriplets()

  def readTriplets(self):
    f = open(self.filename)
    linesRead = 0
    for line in f:
      userid, song, songCount = line.strip().split('\t')

      # Fill in songCounts
      if song in self.songCount:
        self.songCount[song] += 1
      else:
        self.songCount[song] = 1

      # Fill in userHistory
      if userid in self.userHistory:
        self.userHistory[userid][song] = songCount
      else:
        self.userHistory[userid] = { song : songCount }
      linesRead += 1
      if linesRead >= self.numTriplets:
        break

    f.close()

    # re-order songs by popularity
    songsByPopularity = sorted(self.songCount.keys(), key = lambda s: self.songCount[s], \
                               reverse = True)

    print "Done loading %d triplets!" % self.numTriplets

  def recommend(self, newFile):
    """
    Method that, based on the data read in by self.readTriplets() and ## RECOMMENDING ALGORITHM ##
    as defined in #########, will recommend songs to the users featuring in |newFile|
    """
    pass
