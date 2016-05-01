require 'hdf5'
require 'nn'
require 'randomkit'
require 'distributions'

myFile = hdf5.open('../HAR/preprocessed.hdf5','r')
f = myFile:all()
myFile:close()

x_train = f['x_train_with_past']
y_train = f['y_train_with_past']
x_test = f['x_test_with_past']
y_test = f['y_test_with_past']
x_test_withoutpast = f['x_test']
y_test_withoutpast = f['y_test']

model = nn.Sequential()
model:add(nn.Linear(567,300))
model:add(nn.Tanh())
-- model:add(nn.Linear(600,300))
-- model:add(nn.Tanh())
model:add(nn.Linear(300,6))
model:add(nn.LogSoftMax())

model_temp = nn.Sequential()
model_temp:add(nn.Linear(567,300))
model_temp:add(nn.Tanh())
-- model_temp:add(nn.Linear(600,300))
-- model_temp:add(nn.Tanh())
model_temp:add(nn.Linear(300,6))
model_temp:add(nn.LogSoftMax())

parameters, gradParameters = model:getParameters()
parameters_temp, gradParameters_temp = model_temp:getParameters()

criterion = nn.ClassNLLCriterion()

function likelihood(criterion, model, x_train, y_train)
    local pred = model:forward(x_train)
    local nll = criterion:forward(pred, y_train)
    return nll/x_train:size(1)
end

function MH(model, model_temp, criterion, parameters, parameters_temp, x_train, y_train, N)
    local nparam = parameters:size(1)
    local param_prev = torch.Tensor(parameters:size())
    local param_star = torch.Tensor(parameters:size())
    local timer = torch.Timer()
    for i = 1, N do
        p_prev = likelihood(criterion, model, x_train, y_train)
        parameters_temp:copy(parameters + randomkit.normal(torch.Tensor(nparam),0,1))
        p_star = likelihood(criterion, model_temp, x_train, y_train)
        u = randomkit.uniform(0,1)
        if u < torch.exp(-(p_star-p_prev)) then
            parameters:copy(parameters_temp)
        end
        print('Likelihood: '..likelihood(criterion, model, x_train, y_train))
    end
    print('Run-Time: '..timer:time().real)
end

MH(model, model_temp, criterion, parameters, parameters_temp, x_train, y_train, 1)
