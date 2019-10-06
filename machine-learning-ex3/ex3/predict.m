function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ ones(m,1) X];                               % X is m BY 401
Z2m = Theta1 * X' ;       % THETA(1) is 25 BY 401,z2m is (25 x m)
A2m = sigmoid(Z2m); 

% add a0 for the hidden layer ie the bias unit for each of the m % examples

A2m = [ ones(1,m); A2m];                         % a2m is 26 by m
Z3m = Theta2 * A2m;          % THETA(2) is 10by26, a2m is 26 by m
A3m = sigmoid(Z3m);  % a3m is 10 by m so we can transpose to make each row an example
A3m = A3m';                                      % a3m is m by 10
[maxInEachRow, p] = max(A3m,[],2);






% =========================================================================


end
