load UserSongSparseMatrix100000.txt

% R = rating matrix
R = spconvert(UserSongSparseMatrix100000);

% R_max = column vector with max song count for each user
R_max = diag(max(R,[],2).^-1);

% D = normalization matrix
D = diag(sqrt(sum((R).^2,2)).^-1); 

% S = cosine similarity matrix
S = (R_max*D*R)*(R_max*D*R)';