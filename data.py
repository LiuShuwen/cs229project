## We need a data class for this shiz
import numpy as np
import scipy.sparse as sparse

class Data:
    def __init__(self, inputFile, numTriplets, inputTestFile, numTripletsTest):
        self.userIdToIndex = {} # Key: userid, Value: Row in matrix
        self.songIdToIndex = {} # Key: songid, Value: Column in matrix
        self.numUsers = 0
        self.numSongs = 0
        self.numUsersInTraining = 0
        self.numNonZeros = 0
        self.rows = []
        self.columns = []
        self.entries = []
        self.R = None
        self.loadData(inputFile, numTriplets, inputTestFile, numTripletsTest)

    def loadData(self, inputFile, numTriplets, inputTestFile, numTripletsTest):
        """
        Method to load in the data.
        """
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
          if song not in self.songIdToIndex:
            self.songIdToIndex[song] = songIndex
            songIndex += 1

          if userid not in self.userIdToIndex:
            self.userIdToIndex[userid] = userIndex
            userIndex += 1

          # Fill in rows, columns and entries
          rows.append(self.userIdToIndex[userid])
          columns.append(self.songIdToIndex[song])
          entries.append(int(songCount))

          linesRead += 1
          if linesRead >= numTriplets:
            break

        self.numUsersInTraining = userIndex

        f.close()

        # Read in half of the user histories
        linesRead = 0
        f = open(inputTestFile)

        for line in f:
          userid, song, songCount = line.strip().split('\t')

          # Fill in indices
          if song not in self.songIdToIndex:
            self.songIdToIndex[song] = songIndex
            songIndex += 1

          if userid not in self.userIdToIndex:
            self.userIdToIndex[userid] = userIndex
            userIndex += 1

          # Fill in rows, columns and entries
          rows.append(self.userIdToIndex[userid])
          columns.append(self.songIdToIndex[song])
          entries.append(int(songCount))

          linesRead += 1
          if linesRead >= numTriplets:
            break

        self.numSongs = songIndex
        self.numUsers = userIndex
        self.numNonZeros = len(entries)
        self.rows = rows
        self.columns = columns
        self.entries = entries

        self.R = sparse.coo_matrix((entries, (rows, columns)), (self.numUsers, self.numSongs), dtype = np.float)

    def getInfo(self):
        """
        Prints info about the dataset
        """
        print "Number of songs: ", self.numSongs
        print "Number of users: ", self.numUsers
        print "Number of users you need to predict for: ", self.numUsersInTraining
        print "Number of triplets: ", self.numNonZeros
