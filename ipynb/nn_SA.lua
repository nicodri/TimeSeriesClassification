require 'hdf5'
require 'nn'
require 'nngraph'
require 'randomkit'
require 'distributions'

myFile = hdf5.open('../HAR/preprocessed.hdf5','r')
f = myFile:all()
myFile:close()

x = f['x_train_with_past2']
y = f['y_train_with_past']
x_test = f['x_test_with_past2']
y_test = f['y_test_with_past']
x_test_withoutpast = f['x_test']
y_test_withoutpast = f['y_test']

n = x:size(1)
torch.manualSeed(1)
perm = torch.randperm(n):long()
x_train = x:index(1,perm):narrow(1,1,math.floor(0.9*n))
y_train = y:index(1,perm):narrow(1,1,math.floor(0.9*n))
x_train_1 = x_train:narrow(2,1,7)
x_train_2 = x_train:narrow(2,8,561)
x_val = x:index(1,perm):narrow(1,math.floor(0.9*n)+1, n-math.floor(0.9*n))
y_val = y:index(1,perm):narrow(1,math.floor(0.9*n)+1, n-math.floor(0.9*n))
x_val_1 = x_val:narrow(2,1,7)
x_val_2 = x_val:narrow(2,8,561)

-- HELPER (https://gist.github.com/MihailJP/3931841)
function table_copy (t) 
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end

-- Define a model given a parametrisation
function buildmodel(lookup, lt_hid, link1, hid1, activ1, link2, hid2, activ2, link3, activ3)
    
    -- Define default variables
    local lookup = lookup or false
    local lt_hid = lt_hid or 6
    local link1 = link1 or true
    local hid1 = hid1 or 600
    local activ1 = activ1 or true
    local link2 = link2 or false
    local hid2 = hid2 or 300
    local activ2 = activ2 or true
    local link3 = link3 or false
    local activ3 = activ3 or false

    -- Define inputs
    prev_class = nn.Identity()()
    obs = nn.Identity()()

    -- Embed the classes using a lookup table
    if lookup == true then
        prev_ = nn.Narrow(2,1,1)(prev_class)
        prev = nn.View(-1,lt_hid)(nn.LookupTable(6,lt_hid)(prev_))
        len_prev = lt_hid
    else
        prev = nn.Narrow(2,2,6)(prev_class)
        len_prev = 6
    end

    -- Concat the prev class or not ?
    -- Apply a first linear transformation
    if link1 == true then
        layer1 = nn.Linear(561 + len_prev, hid1)(nn.JoinTable(2)({prev,obs}))
    else
        layer1 = nn.Linear(561, hid1)(obs)
    end

    -- Activate output of previous layer using Tanh
    if activ1 == true then
        layer2 = nn.Tanh()(layer1)
    else
        layer2 = layer1
    end

    -- Concat the prev class or not ?
    -- Apply a second linear transformation
    if link2 == true then
        layer3 = nn.Linear(hid1+len_prev, hid2)(nn.JoinTable(2)({prev,layer2}))
    else
        layer3 = nn.Linear(hid1, hid2)(layer2)
    end
    
    -- Activate output of previous layer using Tanh
    if activ2 == true then
        layer4 = nn.Tanh()(layer3)
    else
        layer4 = layer3
    end

    -- Concat the prev class or not ?
    -- Apply a third linear transformation
    if link3 == true then
        layer5 = nn.Linear(hid2+len_prev, 6)(nn.JoinTable(2)({prev,layer4}))
    else
        layer5 = nn.Linear(hid2, 6)(layer4)
    end
    -- Activate output of previous layer using Tanh
    if activ3 == true then
        layer6 = nn.Tanh()(layer5)
    else
        layer6 = layer5
    end

    -- Define output, by taking a logsoftmax on previous output (distribution over the 6 classes)
    out = nn.LogSoftMax()(layer6)

    return nn.gModule({prev_class, obs}, {out})
end

function buildmodel_fromtable(tab)
    return buildmodel(tab[1], tab[2], tab[3], tab[4], tab[5], tab[6], tab[7], tab[8], tab[9], tab[10])
end

-- Assess accuracy of a model input by input
function accuracy(input_1, input_2, output, model)
    local acc = 0.
    for i = 1, input_1:size(1) do
        pred = model:forward({input_1:narrow(1,i,1),input_2:narrow(1,i,1)})
        m, a = pred:max(2)
        if a[1][1] == output[i] then
            acc = acc + 1.
        end
    end
    return acc/input_1:size(1)
end

-- Assess acccuracy on a predicted path
function path_accuracy(pred_path, true_path)
    local acc = 0.
    local path_length = pred_path:size(1)
    for i = 1, path_length do
        if pred_path[i] == true_path[i] then
            acc = acc + 1.
        end
    end
    return acc/path_length
end

 -- Train the model with a SGD
function train_model(train_inputs_1, train_inputs_2, train_outputs, val_inputs_1, val_inputs_2, val_outputs, model, criterion, eta, batch, nEpochs)
    -- Define default variables:
    local batch = batch or 16
    local loss = torch.zeros(nEpochs)
    local av_L = 0
    local f = 0
    local df_do
    local len = train_inputs_1:size(2)
    local ntrain = train_inputs_1:size(1)

    for i = 1, nEpochs do
        -- Display progess
        xlua.progress(i, nEpochs)

        -- timing the epoch
        local timer = torch.Timer()
        av_L = 0
        
        for ii = 1, ntrain, batch do
            
            current_batch_size = math.min(batch,ntrain-ii)
            -- reset gradients
            model:zeroGradParameters()

            -- Forward pass (selection of inputs_batch in case the batch is not full, ie last batch)
            local pred = model:forward({train_inputs_1:narrow(1, ii, current_batch_size),train_inputs_2:narrow(1, ii, current_batch_size)})
            -- Average loss computation
            f = criterion:forward(pred, train_outputs:narrow(1, ii, current_batch_size))
            av_L = av_L + f
            
            -- Backward pass
            df_do = criterion:backward(pred, train_outputs:narrow(1, ii, current_batch_size))
            model:backward({train_inputs_1:narrow(1, ii, current_batch_size),train_inputs_2:narrow(1, ii, current_batch_size)}, df_do)
            model:updateParameters(eta)
            
        end

        loss[i] = av_L/math.floor(ntrain/batch)
        acc_val = accuracy(val_inputs_1, val_inputs_2, val_outputs, model)
        if acc_val > 0.99 then
            return loss[i], acc_val, loss
        end
    end
    return loss[nEpochs], acc_val, loss
    
end

-- Generate new architecture, that is different from the current with probability p
function generate_archi(current, p)
    local archi = table_copy(current)
    local tochange = math.random(10)
    if torch.uniform()<p then
        if type(archi[tochange]) == 'boolean' then
            archi[tochange] = not archi[tochange]
        elseif type(archi[tochange]) == 'number' then
            sgn = math.random(0,1)
            if sgn == 0 then
                sgn = -1
            end
            temp = archi[tochange]+ sgn* math.random(20)
            if temp>6 then
                archi[tochange] = temp
            else
                archi[tochange] = 6
            end
        end
    end
    return archi
end

-- Evalaute cost
function cost(train_acc, val_acc, nparam)
    local train_acc = train_acc
    local val_acc = val_acc
    local nparam = nparam

    return (0*(1-train_acc) + 100000*(1-val_acc) + 0*nparam)/150000
end

-- Finds optimal architecture using Simulated Annealing:
function SA(initial_archi, train_inputs_1, train_inputs_2, train_outputs, val_inputs_1, val_inputs_2, val_outputs, eta, batch, nEpochs, nIt, T, annealing)
    local model_table = {}
    local archi = {}
    local costs = {}
    local train_acc
    local val_acc
    local archi_prev
    local archi_new
    local cost_prev = 0
    local cost_new = 0
    local train_acc_new = 0 
    local acc_val_new = 0
    local criterion = nn.ClassNLLCriterion()
    local model_prev
    local ct = 0

    model_table[1] = buildmodel_fromtable(initial_archi)
    archi[1] = initial_archi

    train_acc, val_acc = train_model(train_inputs_1, train_inputs_2, train_outputs, val_inputs_1, val_inputs_2, val_outputs, model_table[1], criterion, 0.01, batch, nEpochs)
    costs[1] = cost(train_acc, val_acc, model_table[1]:getParameters():size(1))

    archi_prev = table_copy(initial_archi)
    model_prev = model_table[1]:clone()
    cost_prev = costs[1]
    print(cost_prev)
    for i = 2,nIt do
        archi_new = generate_archi(archi_prev, 0.5)
        model_new = buildmodel_fromtable(archi_new) 
        train_acc_new, val_acc_new = train_model(train_inputs_1, train_inputs_2, train_outputs, val_inputs_1, val_inputs_2, val_outputs, model_new, criterion, eta, batch, nEpochs)
        cost_new = cost(train_acc_new, val_acc_new, model_new:getParameters():size(1))

        if torch.uniform() < torch.exp(-(cost_new-cost_prev)/T) then
            archi_prev = table_copy(archi_new)
            model_prev = model_new:clone()
            cost_prev = cost_new
            ct = ct + 1
            if ct == 3 then
                T = T * annealing
                ct = 0
            end
        end
        print(cost_prev)
        model_table[i] = model_prev:clone()
        costs[i] = cost_prev
        archi[i] = table_copy(archi_prev)
    end

    return archi, costs, model_table
end

--buildmodel(lookup, lt_hid, link1, hid1, activ1, link2, hid2, activ2, link3, activ3)
-- gmod = buildmodel(true, 10, true, 300, true, false, 300, false, false, false)
-- test = gmod:forward({x_train_1:narrow(1,1,1), x_train_2:narrow(1,1,1)})
-- print(accuracy(x_train_1:narrow(1,1,1), x_train_2:narrow(1,1,1),y_train:narrow(1,1,1), gmod))
criterion = nn.ClassNLLCriterion()

-- train_model(train_inputs_1, train_inputs_2, train_outputs, val_inputs_1,
--                     val_inputs_2, val_outputs, model, criterion, eta, nEpochs, batch)
-- test = train_model(x_train_1, x_train_2, y_train, x_val_1, x_val_2, y_val, gmod, criterion, 0.01, 20, 16)

T = 1
annealing = 0.8
nIt = 20
nEpochs = 7
batch = 16
eta = 0.01
initial_archi = {true, 10, true, 300, true, true, 300, true, true, true}


archi, costs, model_table = SA(initial_archi, x_train_1, x_train_2, y_train, x_val_1, x_val_2, y_val, eta, batch, nEpochs, nIt, T, annealing)





