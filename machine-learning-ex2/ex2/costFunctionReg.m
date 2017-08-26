function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters. 
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost of a particular choice of theta.
  %               You should set J to the cost.
  %               Compute the partial derivatives and set grad to the partial
  %               derivatives of the cost w.r.t. each parameter in theta
  
  
  % Compute the cost
  h_theta_x = sigmoid(X * theta);
  % not regularizing theta(1)
  reg_term = (sum(theta .^ 2) - (theta(1) ^ 2)) * lambda / (2*m); 
  J = ( sum(y .* log(h_theta_x) + (1-y) .* log(1-h_theta_x)) / (-m) ) + reg_term;
  
  
  % Compute the gradient
  delta = [];
  % storing the difference between hypothesis output and real value
  difference = h_theta_x - y; 
  delta = repmat(difference, 1, size(X, 2)); % delta is now a mxm vectors
  
  % perform the summation and term-by-term multiplication
  theta_backup = theta;
  theta_backup(1) = 0; % not regularizing theta(1)
  grad = ( sum(X .* delta, 1) / m ) + (theta_backup * lambda / m)';
  
  
  % =============================================================
  
end
