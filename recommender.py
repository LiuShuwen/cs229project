import numpy as np
import scipy.sparse as sparse

class Recommender():
  def __init__(self, filename, numTriplets = 1000):
    self.filename = filename
    self.numTriplets = numTriplets
    self.songCount = {} # map song to number of times it's been listened to
    self.userHistory = {}
    self.userSongMatrix = None # Row: userid, Column: Songid, Entries: Songcount
    self.userIdToIndex = {} # Key: userid, Value: Row in matrix
    self.songIdToIndex = {} # Key: songid, Value: Column in matrix
    self.numSongs = 0
    self.numUsers = 0

    self.readTriplets()

  def readTriplets(self):
    userIndex = 0
    songIndex = 0
    rows = []
    columns = []
    entries = []
    linesRead = 0
    f = open(self.filename)

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

      # Fill in matrix
      if song not in self.songIdToIndex:
        self.songIdToIndex[song] = songIndex
        songIndex += 1

      if userid not in self.userIdToIndex:
        self.userIdToIndex[userid] = userIndex
        userIndex += 1

      rows.append(self.userIdToIndex[userid])
      columns.append(self.songIdToIndex[song])
      entries.append(songCount)

      linesRead += 1
      if linesRead >= self.numTriplets:
        break

    self.numSongs = songIndex + 1
    self.numUsers = userIndex + 1

    self.userSongMatrix = sparse.coo_matrix( (entries, (rows,columns)), shape=(self.numUsers, self.numSongs) )

    f.close()

    # re-order songs by popularity
    songsByPopularity = sorted(self.songCount.keys(), key = lambda s: self.songCount[s], \
                               reverse = True)

    print "Done loading %d triplets!" % self.numTriplets

  def GetCosineSimilarityMatrix(self):
    """
    return Cosine Similarity Matrix where rating for a song is defined as songCount/maxSongCount
    """
    #maxSongCount1 = max( self.userHistory[userid1][song] for song in self.userHistory[userid1])
    #maxSongCount2 = max( self.userHistory[userid2][song] for song in self.userHistory[userid2])

    #userHistoryIntersection = {}
    pass

  def countToRating(self, userid):
    pass

  def recommend(self, newFile):
    """
    Method that, based on the data read in by self.readTriplets() and ## RECOMMENDING ALGORITHM ##
    as defined in #########, will recommend songs to the users featuring in |newFile|
    """
    pass
