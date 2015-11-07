import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

def meanAveragePrecision(M, predictions, verbose = False):
    """
    Given a binary valued sparse feedback matrix |M|, indicating whether user u listened to
    song i at index (u,i), and a predicted ranking of songs for each user |prediction|,
    returns the mean average precision metric of the prediction.
    Assume: prediction is a matrix of size #users * #predictions (represented by tau in
    literature). Each row u is therefore the song ID predictions for user u, in ascending
    order of ranking
    """
    users, tau = predictions.shape
    mAP = 0 # mean average precision across all users
    for user in range(users):
        # check whether first recommended song for user is in his/her listening history
        predictionsAtK_1 = M[user, predictions[user][0]]
        precision = M[user, predictions[user][0]]
        positiveExamples = M[user, predictions[user][0]]
        for k in range(1,tau):
            predictionsAtK = (k/float(k+1))*predictionsAtK_1 + \
                               (1/float(k+1))*M[user, predictions[user][k]]
            predictionsAtK_1 = predictionsAtK
            # if the k-th recommended song for the user appears in listening history,
            # increment average precision
            if M[user, predictions[user][k]]:
                precision += predictionsAtK
                positiveExamples += 1
        if positiveExamples:
            # divide by tau, or number of positive recommendations??
            mAP += precision/float(positiveExamples)
        
        if verbose:
            print "Done with user %d; precision: %s " % (user, precision)
    mAP /= users
    
    return mAP

# Create canonical user indexing for test set
usersIndexing, userIdx = {}, 0
f = open('../../data/eval/year1_test_triplets_visible.txt', 'r')

for line in f:
    userId, song, _ = line.strip().split('\t')
    if not userId in usersIndexing:
        usersIndexing[userId] = userIdx
        userIdx += 1

f.close()

# Create canonical indexing of songs for entire data set
songsIndexing, songIdx = {}, 0
f = open('../../data/train_triplets.txt')
for line in f:
    _, songId, _ = line.strip().split('\t')
    if not songId in songsIndexing:
        songsIndexing[songId] = songIdx
        songIdx += 1
f.close()

# Step 1: load the triplets and compute song counts
with open('../../data/eval/year1_test_triplets_visible.txt', 'r') as f:
    song_to_count = dict() 
    for line in f:
        _, song, _ = line.strip().split('\t') 
        if song in song_to_count: 
            song_to_count[song] += 1 
        else: 
            song_to_count[song] = 1 
            pass
        pass
    pass

# Step 2: sort by popularity
songs_ordered = sorted( song_to_count.keys(), 
                        key=lambda s: song_to_count[s],
                        reverse=True)

# Step 3: load the visible user histories
with open('../../data/eval/year1_test_triplets_visible.txt', 'r') as f:
    user_to_songs = dict() 
    for line in f:
        user, song, _ = line.strip().split('\t') 
        if user in user_to_songs: 
            user_to_songs[user].add(song) 
        else: 
            user_to_songs[user] = set([song])
            pass
        pass
    pass

# Step 4: load the hidden user histories
with open('../../data/eval/year1_test_triplets_hidden.txt', 'r') as f:
    user_to_songs_hidden = dict() 
    for line in f:
        user, song, _ = line.strip().split('\t') 
        if user in user_to_songs_hidden: 
            user_to_songs_hidden[user].add(song) 
        else: 
            user_to_songs_hidden[user] = set([song])
            pass
        pass
    pass

numUsers = len(usersIndexing)

entries, rows, columns = [], [], []

for user_i, user in enumerate(usersIndexing):
    for song in user_to_songs_hidden[user]:
        if song in songsIndexing:
            rows.append(user_i)
            columns.append(songsIndexing[song])
            entries.append(1)

userSongMatrix = sparse.coo_matrix( (entries, (rows,columns)), \
                                   shape=(numUsers, 400000) )

predictions = np.zeros([numUsers, 500])

# Step 5: generate the prediction - so far, just recommending top 500 songs to everyone
for user_i, user in enumerate(usersIndexing):
    for recommendation_j, song in enumerate(songs_ordered[:500]):
        if not song in user_to_songs[user]:
            predictions[user_i][recommendation_j] = songsIndexing[song]
            
# Step 6: evaluate mAP
feedbackMatrix = userSongMatrix.tocsr()
mAP = meanAveragePrecision(feedbackMatrix, predictions)
print "Mean Average Precision: %s" % mAP