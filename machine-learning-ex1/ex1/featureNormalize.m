function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.
%
% Hint: You might find the 'mean' and 'std' functions useful.
%

num_col = size(X,2); % number of features of the dataset

% Compute the mean and standard deviation
for i = 1:num_col

  mu(i) = mean(X(:,i)); % compute and store the mean
  X_norm(:,i) -= mu(i); % subtract the mean from the dataset

  sigma(i) = std(X(:,i)); % compute and store the standard deviation
  X_norm(:,i) /= sigma(i); % divide each feature by it's standard deviation

endfor

% fprintf('Print the matrix mu\n');
% disp(mu);
%
%
% fprintf('Print the matrix sigma\n');
% disp(sigma);


% ============================================================

end
