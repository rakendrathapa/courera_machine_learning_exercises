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
list_val = [0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0];
err_mean = 1.0;     % MAX error value between y and yval
C_sigma_list = [0 0 0];

for i =1:length(list_val),
  C = list_val(i);
  for j=1:length(list_val),
    sigma = list_val(j);
    %fprintf('C=%f sigma=%f\n', C, sigma);
    
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predictions = svmPredict(model, Xval);
    
    prediction_err_mean = mean(double(predictions ~= yval));
    %fprintf('prediction_err_mean=%f err_mean=%f\n', prediction_err_mean, err_mean);
    
    if prediction_err_mean <= err_mean,
      %fprintf('prediction_err_mean < err_mean\n');
      err_mean = prediction_err_mean;
      C_val = C;
      sigma_val = sigma;
      
      C_sigma_list = [C_sigma_list;prediction_err_mean C_val sigma_val];
    end;    
  end;
end;

%C_sigma_list(2:end, :)
C = C_val;
sigma = sigma_val;
% =========================================================================

end
