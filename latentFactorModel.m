%% Reading in data

clear all

load UserSongSparseMatrix1000.txt

% Count = count matrix
% Row: User
% Column: Song
% Entries: SongCount
% Size: numUsers * numSongs
Count = spconvert(UserSongSparseMatrix1000);

numUsers = size(Count,1); numSongs = size(Count,2);

% InvMaxSongCount = diagonal matrix with inverse of max song count for each user
% Size: numUsers * numUsers
InvMaxSongCount = diag(max(Count,[],2).^-1);

% R = rating matrix.
% For each user (row), we divide the row by the highest song
% count. Therefore, instead of song counts, we have
% a number between 0 and 1, which we use as a rating.
% Row: User
% Column: Song
% Entries: SongCount/max(SongCount for each user)
% Size: numUsers * numSongs
R = InvMaxSongCount*Count;
R = R'; % Use transposed matrix for LFM

%% Training
% Minimize sum of squared errors with regularization

% Parameters
k = 3; % Number of factors
lambda1 = 1; % Regularization for Q
lambda2 = 1; % Regularization for P
eta1 = 0.01; % Learning rate for Q
eta2 = 0.01; % Learning rate for P
Q = rand(numSongs,k);
P = rand(numUsers,k);
maxNumIters = 1;

% You need to run this until convergence, but right now just run for a few
% iterations
for i = 1:maxNumIters
    % Stochastic gradient descent
    E = 2*(R - Q*P');
    Q = Q + eta1*(E*P - lambda1*Q);
    P = P + eta2*(E'*Q - lambda2*P);
end

display(sum(sum(R - Q*P'))/sum(sum(R)))





