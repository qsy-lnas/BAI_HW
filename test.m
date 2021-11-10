x = [1 -1 2; 1 0 1; 1 2 -1; 1 1 0];
Y = [1.3 -0.5 2.6 0.9];
Y = Y.';
I = eye(3);
lambda = 10;
beta = (x.'*x + lambda * I)\( x.' * Y)