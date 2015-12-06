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
        self.C = None # Counts matrix
        self.R = None # Rating matrix
        self.C_hidden = None
        self.loadData(inputTrainingFile, numTriplets, inputTestFile, inputHiddenTestFile, numTripletsTest)
        self.setRatingType()

    def loadData(self, inputTrainingFile, numTriplets, inputTestFile, inputHiddenTestFile, numTripletsTest):
        """
        Method to load in the data.
        Loads in the training set and the visible half of playlist into Matrix C.
        Loads in the hidden half of the playlist into Matrix C_hidden
        """
        userIndex = 0
        songIndex = 0
        rows = []
        columns = []
        entries = []

        maxLines = numTriplets

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
              if linesRead >= maxLines:
                break

            if inputFile == inputTrainingFile:
                self.numUsersInTraining = userIndex
                maxLines = numTripletsTest

            if inputFile == inputTestFile:
                self.numSongs = songIndex
                self.numUsers = userIndex
                self.numNonZeros = len(entries)
                self.rows = rows
                self.columns = columns
                self.entries = entries
                self.C = sparse.coo_matrix((entries, (rows, columns)), (self.numUsers, self.numSongs), dtype = np.float).tocsr()
                # reset everything to zero to read in the hidden matrix
                rows = []
                columns = []
                entries = []

            if inputFile == inputHiddenTestFile:
                self.numSongsUnseen = songIndex - self.numSongs
                self.C_hidden = sparse.coo_matrix((entries, (rows, columns)), (userIndex, songIndex), dtype = np.float).tocsr()

            f.close()

    def setRatingType(self, ratingType=1):
        """
        Transform R to a matrix of "ratings" rather than song counts.
        Type parameters determines how this is done:
        0 : Song counts
        1 : Divide each user count by that users maximum count
        2 : Normalize user counts (i.e. divide each entry by sum of this user's counts)
        """
        if ratingType == 0:
            self.R = self.C.tocsc()
        if ratingType == 1:
            maxVec = self.C.max(axis=1).transpose()
            invMaxVec = 1./maxVec.todense()
            maxDiag = sparse.diags(invMaxVec.tolist()[0], 0)
            self.R = maxDiag * self.C
            self.R = self.R.tocsc()

        if ratingType == 2:
            sumVec = self.C.sum(axis=1).transpose()
            invSumVec = 1./sumVec
            sumDiag = sparse.diags(invSumVec.tolist()[0], 0)
            self.R = sumDiag * self.C
            self.R = self.R.tocsc()

    def getInfo(self):
        """
        Prints info about the dataset
        """
        print "Number of songs: ", self.numSongs
        print "Number of users: ", self.numUsers
        print "Number of users you need to predict for: ", self.numUsers - self.numUsersInTraining
        print "Number of songs that have never been seen in training: ", self.numSongsUnseen
        print "Number of triplets: ", self.numNonZeros
    
    def averagePrecision(self, user, predictions, k = 500):
        """
        Computes the average precision at k for |user|.
        |predictions| should be a list of song ID's
        """
        score, numHits = 0., 0.
        numHidden = (self.C_hidden.indptr[user + 1] - self.C_hidden.indptr[user])
        
        for i, p in enumerate(predictions):
            # check if p is actually present
            if self.C_hidden[user, p] > 0:
                numHits += 1.
                score += numHits/(i + 1.)
            
        return score/min(numHidden, k)