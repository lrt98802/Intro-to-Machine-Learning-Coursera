function [C, sigma] = dataset3Params(X, y, Xval, yval)
  %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
  %where you select the optimal (C, sigma) learning parameters to use for SVM
  %with RBF kernel
  %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
  %   sigma. You should complete this function to return the optimal C and 
  %   sigma based on a cross-validation set.
  %
  
  % You need to return the following variables correctly.
  C = 1;
  sigma = 0.3;
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Fill in this function to return the optimal C and sigma
  %               learning parameters found using the cross validation set.
  %               You can use svmPredict to predict the labels on the cross
  %               validation set. For example, 
  %                   predictions = svmPredict(model, Xval);
  %               will return the predictions on the cross validation set.
  %
  %  Note: You can compute the prediction error using 
  %        mean(double(predictions ~= yval))
  %
  
  list = [0.01 ; 0.03 ; 0.1 ; 0.3 ; 1 ; 3 ; 10 ; 30];
  % initialize the optimal_error as an arbitrary large number
  optimal_error = 100; 
  
  for i = 1:size(list, 1)
    C_temp = list(i);
    
    for j = 1:size(list, 1)
      sigma_temp = list(j);
      
      % Train the model using the training set
      model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp)); 
      % generate predictions using the cross-validation set
      predictions = svmPredict(model, Xval);
      % compute the error
      error = mean(double(predictions ~= yval));
      
      % check if the error is a new optimal 
      if( error < optimal_error )
        optimal_error = error;
        C = C_temp;
        sigma = sigma_temp;
      endif
      
    endfor
  endfor
  
  
  
  fprintf('Optimal C is: %f\n', C);
  fprintf('Optimal sigma is: %f\n', sigma);
  
  
  % =========================================================================
  
end
