%% Reading in data

load UserSongSparseMatrix100000.txt

% Count = count matrix
% Row: User
% Column: Song
% Entries: SongCount
% Size: numUsers * numSongs
Count = spconvert(UserSongSparseMatrix100000);

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
Rating = InvMaxSongCount*Count;

%% Computing Cosine Similarity

% UserNormalize = diagonal user normalization matrix
% Entries: Inverse of norm of row for each user
% Size: numUsers * numUsers
UserNormalize = diag(sqrt(sum((Rating).^2,2)).^-1); 

% CosineUser = user-user cosine similarity matrix
% Entries (i,j): cosine similarity between user i and j
% Size: numUsers * numUsers
CosineUser = (UserNormalize*Rating)*(UserNormalize*Rating)';