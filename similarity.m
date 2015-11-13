%% Reading in data

load UserSongSparseMatrix10000.txt

% Count = count matrix
% Row: User
% Column: Song
% Entries: SongCount
% Size: numUsers * numSongs
Count = spconvert(UserSongSparseMatrix10000);

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
Rating = InvMaxSongCount*Count;

%% Cosine Similarity User/User

% UserNormalize = diagonal user normalization matrix
% Entries: Inverse of norm of row for each user
% Size: numUsers * numUsers
UserNormalize = diag(sqrt(sum((Rating).^2,2)).^-1); 

% CosineUser = user-user cosine similarity matrix
% Entries (i,j): cosine similarity between user i and j
% Size: numUsers * numUsers
CosineUser = (UserNormalize*Rating)*(UserNormalize*Rating)';

% Create logical array of counts
BinaryCount = logical(Count);

% Calculate Score
% Each row is the score user would give to the different songs.
% Pick the e.g. the 20 highest score => recommendations
Score = CosineUser*BinaryCount;

% Display the 5 top songs for user 1
[sortedValues,sortIndex] = sort(Score(1,:),'descend');
sortIndex(1:5)

%% Cosine Similarity Song/Song

% SongNormalize = diagonal song normalization matrix
% Entries: Inverse of norm of column for each user
% Size: numSongs * numSongs
SongNormalize = diag(sqrt(sum((Rating).^2,1)).^-1);

% CosineUser = user-user cosine similarity matrix
% Entries (i,j): cosine similarity between user i and j
% Size: numUsers * numUsers
CosineSong = (Rating*SongNormalize)'*(Rating*SongNormalize);