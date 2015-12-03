## We need a data class for this shiz
import numpy as np
import scipy.sparse as sparse

class Data:
    def __init__(self, inputTrainingFile, numTriplets, inputTestFile, inputHiddenTestFile, numTripletsTest):
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
        self.R_hidden = None
        self.loadData(inputTrainingFile, numTriplets, inputTestFile, inputHiddenTestFile, numTripletsTest)

    def loadData(self, inputTrainingFile, numTriplets, inputTestFile, inputHiddenTestFile, numTripletsTest):
        """
        Method to load in the data.
        Loads in the training set and the visible half of playlist into Matrix R.
        Loads in the hidden half of the playlist into Matrix R_hidden
        """
        userIndex = 0
        songIndex = 0
        rows = []
        columns = []
        entries = []

        for inputFile in [inputTrainingFile, inputTestFile, inputHiddenTestFile]:
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

            if inputFile == inputTrainingFile:
                self.numUsersInTraining = userIndex

            if inputFile == inputTestFile:
                self.numSongs = songIndex
                self.numUsers = userIndex
                self.numNonZeros = len(entries)
                self.rows = rows
                self.columns = columns
                self.entries = entries
                self.R = sparse.coo_matrix((entries, (rows, columns)), (self.numUsers, self.numSongs), dtype = np.float)
                # reset everything to zero to read in the hidden matrix
                rows = []
                columns = []
                entries = []

            if inputFile == inputHiddenTestFile:
                self.R_hidden = sparse.coo_matrix((entries, (rows, columns)), (userIndex, songIndex), dtype = np.float)

            f.close()

    def getInfo(self):
        """
        Prints info about the dataset
        """
        print "Number of songs: ", self.numSongs
        print "Number of users: ", self.numUsers
        print "Number of users you need to predict for: ", self.numUsersInTraining
        print "Number of triplets: ", self.numNonZeros
