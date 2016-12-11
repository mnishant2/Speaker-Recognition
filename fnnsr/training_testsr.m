% this is the main file for training of Autonomous car

% Load Training Data
fprintf('Loading Training DATA...\n');

load('inputlabel1k.mat');
load('inputdata1k.mat');
trainingdata=[inputdata inputlabel];


fprintf('Applying backpropagation...');

a=randperm(size(trainingdata,1));
X=double(trainingdata(a,:));
x_train=X(1:750,1:13);
x_test=X(751:1000,1:13);
y_train=X(1:750,14);
y_test=X(751:1000,14);
% [X,mu,sigma]=featureNormalize(X);

   % imshow(reshape(data(:,i),176,144),[]);
    %pause(0.001);

    
hidden_layer1_size=64;
% hidden_layer2_size=20;
% hidden_layer3_size=15;
num_labels=10;
input_layer_size=size(x_train,2);


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer1_size);
initial_Theta2 = randInitializeWeights(hidden_layer1_size,num_labels);
%initial_Theta3 = randInitializeWeights(hidden_layer2_size, hidden_layer3_size);
% initial_Theta3 = randInitializeWeights(hidden_layer2_size, num_labels);


% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%==================backpropagation==============================
options = optimset('MaxIter',1000);

%  You should also try different values of lambda
lambda = 0.3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p)nnmy(p, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   num_labels, x_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)  
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):(0 + (hidden_layer1_size * (input_layer_size + 1)))+ num_labels*(hidden_layer1_size+1)), ...
                  num_labels, (hidden_layer1_size + 1));
             
% Theta3 = reshape(nn_params((1 + (hidden_layer2_size * (hidden_layer1_size + 1))):(0 + (hidden_layer2_size * (hidden_layer1_size + 1)))+num_labels*(hidden_layer2_size+1)), ...
%                  num_labels, (hidden_layer2_size + 1));
             
% Theta4 =  reshape(nn_params((1 + (hidden_layer3_size * (hidden_layer2_size + 1))):(0 + (hidden_layer3_size * (hidden_layer2_size + 1)))+num_labels*(hidden_layer3_size+1)), ...
%                  num_labels, (hidden_layer3_size + 1));

save('neural_param.mat','Theta1','Theta2');
fprintf('end of training\n');
pred=predictsr(Theta1,Theta2,x_test,y_test);
size(pred')
pred=pred-1;
z=(y_test-pred'==0);

k=mean(z)*100;
sprintf('Training acuracy is:%f percent',k)