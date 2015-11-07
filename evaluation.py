import numpy as np
from scipy.sparse import csr_matrix

def meanAveragePrecision(M, predictions):
    """
    Given a binary valued sparse feedback matrix |M|, indicating whether user u listened to
    song i at index (u,i), and a predicted ranking of songs for each user |prediction|,
    returns the mean average precision metric of the prediction.
    Assume: prediction is a matrix of size #users * #predictions (represented by tau in
    literature). Each row u is therefore the song ID predictions for user u, in ascending
    order of ranking
    """
    users, tau = predictions.shape
    predictionsAtK = np.zeros(predictions.shape)
    mAP = 0 # mean average precision across all users
    for user in range(users):
        # check whether first recommended song for user is in his/her listening history
        predictionsAtK[user][0] = M[user, predictions[user][0]]
        averagePrecision = M[user, predictions[user][0]]
        positiveExamples = M[user, predictions[user][0]]
        for k in range(1,tau):
            predictionsAtK[user][k] = (k/float(k+1))*predictionsAtK[user][k-1] + \
                                      (1/float(k+1))*M[user, predictions[user][k]]
            # if the k-th recommended song for the user appears in listening history,
            # increment average precision
            if M[user, predictions[user][k]]:
                averagePrecision += predictionsAtK[user][k]
                positiveExamples += 1
        mAP += averagePrecision/float(positiveExamples)
    
    mAP /= users
    
    print predictionsAtK, mAP
    
    return mAP