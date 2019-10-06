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

z = X * theta;
var1 = sigmoid(z);
var2 = -(y')*log(var1) - (1-y')*log(1-var1);

% for Regularization we dont want to penalize theta(1)
thetaReg = theta(2:length(theta));

% Since var2 is already a scalar value we dont need 1./m but be carefull


J = 1/m *var2 + lambda/(2*m) * sum(thetaReg.^2);

% Now gradient
var3 = sigmoid(X * theta) - y;

var4 = var3' * X;
gradWithoutReg = 1/m * var4';

% to add the reg value we need to modify the theta[0; thetaReg]
% so it contains [theta(1)= 0; theta(2);theta(3) ... theta(n+1)]
% lambda/m *theta(j) for all j >=1
thetaGrad = [ 0; thetaReg];

grad = gradWithoutReg + lambda/m *thetaGrad;

% grad(1) = gradWithoutReg(1);




% =============================================================

end
