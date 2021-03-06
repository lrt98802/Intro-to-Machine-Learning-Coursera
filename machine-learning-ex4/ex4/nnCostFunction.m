function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Compute the Cost with regularization

% ====== First/Input Layer =======
% Add bias units to input matrix X
X = [ones(m,1) X];



% ====== Second/Hidden Layer =======
% Compute the activation units of the second layer
z_2 = X * Theta1';
a_2 = sigmoid(z_2); % apply the sigmoid function

% Add bias units to activation units of the second layer
a_2 = [ones(size(a_2, 1), 1) a_2];



% ====== Third/Output Layer =======
% Compute the activation units of the third layer
z_3 = a_2 * Theta2';
h_theta = sigmoid(z_3); % Apply the sigmoid function

% Generate the y matrix where each row represents a 0/1 vector 
yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;
endfor

% Compute the cost function with double summation
J = sum(sum(yVec .* log(h_theta) + (1-yVec) .* log(1-h_theta))) / (-m);

% Perform the regularization
theta1_reg = Theta1(:, 2:size(Theta1,2)); % avoid regularizing the bias units
theta2_reg = Theta2(:, 2:size(Theta2,2));

reg_term = ( sum(sum(theta1_reg .^ 2),2) + sum(sum(theta2_reg .^ 2),2) ) / (2*m) * lambda ;

J = J + reg_term;



% Compute the gradient

% Initialize the capital delta for all layers (except the output layer)
capital_Delta_1 = zeros(hidden_layer_size, input_layer_size+1);
capital_Delta_2 = zeros(num_labels, hidden_layer_size+1);


% Loop through all the input examples
for i = 1:m
  
  % ====== First/Input Layer: Forward propagation =======
  a_sub_1 = X(i,:);  % Set the values of a(1) to the ith training example
  
  
  % ====== Second/Hidden Layer: Forward propagation =======
  z_sub_2 = a_sub_1 * Theta1';
  a_sub_2 = sigmoid(z_sub_2);
  a_sub_2 = [1 a_sub_2]; % add bias unit
  
  % ====== Third/Output Layer: Forward propagation =======
  z_sub_3 = a_sub_2 * Theta2';
  a_sub_3 = sigmoid(z_sub_3); % this represents the output h_theta(x)
  
  % generate the y vector
  y_out = zeros(num_labels, 1);
  number = y(i); 
  y_out(number) = 1;
  
  % Start back propagation
  delta_3 = a_sub_3' - y_out; % vector delta_3 is of dimension num_labels
  
  % ====== Second/Hidden Layer: Back propagation =======
  delta_2 =  Theta2' * delta_3 .* [1, sigmoidGradient(z_sub_2)]'; 
  % remove the bias unit
  delta_2 = delta_2(2:end); % vector delta_2 is of dimension hidden_layer_size
  
  % add up the value of capital delta for all layers (except the output layer)
  capital_Delta_1 =  capital_Delta_1 + (delta_2 * a_sub_1);
  capital_Delta_2 = capital_Delta_2 + (delta_3 * a_sub_2);
  
  
endfor

% Obtain the unregularized gradient 
Theta1_grad = capital_Delta_1 ./ m;
Theta2_grad = capital_Delta_2 ./ m;


% Perform the regularization
theta1_reg = [zeros(hidden_layer_size, 1) theta1_reg];
Theta1_grad += theta1_reg .* lambda ./ m;

theta2_reg = [zeros(num_labels, 1) theta2_reg];
Theta2_grad += theta2_reg .* lambda ./ m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
