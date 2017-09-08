function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  %regression with multiple variables
  %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  %   cost of using theta as the parameter for linear regression to fit the 
  %   data points in X and y. Returns the cost in J and the gradient in grad
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost and gradient of regularized linear 
  %               regression for a particular choice of theta.
  %
  %               You should set J to the cost and grad to the gradient.
  %
  
  % Compute the regularized costs
  difference = (X * theta) - y;
  cost = sum( difference .^ 2 ) / (2*m);
  reg_term = ( sum(theta .^ 2) - (theta(1) ^ 2) ) * lambda / (2*m);
  
  J = cost + reg_term; % Assign the costs
  
  
  % Compute the gradient
  delta = [];
  delta = repmat(difference, 1, size(X, 2)); % delta is now a mxm vectors
  
  % perform the summation and term-by-term multiplication
  theta_backup = theta;
  theta_backup(1) = 0; % not regularizing theta(1)
  grad = (sum(X .* delta, 1) / m)' + (theta_backup * lambda / m);
  
  
  
  
  
  % =========================================================================
  
  grad = grad(:);
  
end
