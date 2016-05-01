require 'hdf5'
require 'nn'
require 'randomkit'

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

function train_model(train_inputs, train_outputs, test_inputs, test_outputs, model, criterion, eta, nEpochs, batch)
    -- Train the model with a SGD
    -- standard parameters are
    -- nEpochs = 1
    -- eta = 0.01

    -- To store the loss
    local batch = batch or 16
    local loss = torch.zeros(nEpochs)
    local av_L = 0
    local f = 0
    local df_do
    local len = train_inputs:size(2)
    for i = 1, nEpochs do
        -- Display progess
        xlua.progress(i, nEpochs)

        -- timing the epoch
        local timer = torch.Timer()
        av_L = 0
        
        for ii = 1, train_inputs:size(1), batch do
            
            current_batch_size = math.min(batch,train_inputs:size(1)-ii)
            -- reset gradients
            model:zeroGradParameters()

            -- Forward pass (selection of inputs_batch in case the batch is not full, ie last batch)
            local pred = model:forward(train_inputs:narrow(1, ii, current_batch_size))
            -- Average loss computation
            f = criterion:forward(pred, train_outputs:narrow(1, ii, current_batch_size))
            av_L = av_L + f
            
            -- Backward pass
            df_do = criterion:backward(pred, train_outputs:narrow(1, ii, current_batch_size))
            model:backward(train_inputs:narrow(1, ii, current_batch_size), df_do)
            model:updateParameters(eta)
            
        end

        loss[i] = av_L/math.floor(train_inputs:size(1)/batch)
        acc_test = accuracy(test_inputs, test_outputs, model)
        print('Epoch '..i..': '..timer:time().real)
        print('\n')
        print('Average Loss: '.. loss[i])
        print('\n')
        print('Accucary on test: '.. acc_test)
        print('***************************************************')
        if acc_test > 0.99 then
            break
        end
    end

    return loss
end

function accuracy(input, output, model)
    local acc = 0.
    for i = 1, input:size(1) do
        pred = model:forward(input[i])
        m, a = pred:view(6,1):max(1)
        if a[1][1] == output[i] then
            acc = acc + 1.
        end
    end
    return acc/input:size(1)
end

function compute_logscore(inputs, i, model, C)
    local y = torch.zeros(C,C)
    local hot_1 = torch.zeros(C)
    for j = 1, C do
        hot_1:zero()
        hot_1[j] = 1
        y:narrow(1,j,1):copy(model:forward(torch.cat(hot_1,inputs[i],1)))
    end
    return y
end

-- Evaluates the highest scoring sequence:
function viterbi(inputs, init, compute_logscore, model, C)
    
    local y = torch.zeros(C,C)
    -- Formating tensors
    local initial = torch.zeros(C, 1)
    -- initial started with a start of sentence: <t>

    initial[{init,1}] = 1
    initial:log()

    -- number of classes
    local n = inputs:size(1)
    local max_table = torch.Tensor(n, C)
    local backpointer_table = torch.Tensor(n, C)
    -- first timestep
    -- the initial most likely paths are the initial state distribution
    local maxes, backpointers = (initial + compute_logscore(inputs, 1, model, C)[init]):max(2)
    max_table[1] = maxes
    -- remaining timesteps ("forwarding" the maxes)
    for i=2,n do
        -- precompute edge scores
       
        y:copy(compute_logscore(inputs, i, model, C))
        scores = y:transpose(1,2) + maxes:view(1, C):expand(C, C)

        -- compute new maxes 
        maxes, backpointers = scores:max(2)

        -- record
        max_table[i] = maxes
        backpointer_table[i] = backpointers
    end
    -- follow backpointers to recover max path
    local classes = torch.Tensor(n)
    maxes, classes[n] = maxes:max(1)
    for i=n,2,-1 do
        classes[i-1] = backpointer_table[{i, classes[i]}]
    end

    return classes
end

function compute_fscore(predicted_classes, true_classes)
    print('here')
    local n = predicted_classes:size(1)
    local right_pred = 0
    local positive_true = 0
    local positive_pred = 0
    for i=1,n do
        if predicted_classes[i] > 1 then
            positive_pred = positive_pred + 1
        end
        if true_classes[i] > 1 then
            positive_true = positive_true + 1
        end
        if (true_classes[i] == predicted_classes[i]) and true_classes[i] > 1 then
            right_pred = right_pred + 1
        end
    end
    local precision = right_pred/positive_pred
    local recall = right_pred/positive_true
    local fscore = 2*precision*recall/(precision+recall)
    print('Precision: ' .. precision )
    print('Recall: ' ..  recall)
    print('F-score: ' ..  fscore)
    return fscore
end        

parameters, gradParameters = model:getParameters()
print(parameters:size())

torch.manualSeed(0)
randomkit.uniform(parameters,-0.05,0.05)

criterion = nn.ClassNLLCriterion()

loss = train_model(x_train, y_train, x_test, y_test, model, criterion, 0.01, 20, 16)

input_test = x_test_withoutpast:narrow(1,2,x_test_withoutpast:size(1)-1)

predicted_test = viterbi(input_test, y_test_withoutpast[1], compute_logscore, model, 6)

fscore = compute_fscore(predicted_test, y_test)
