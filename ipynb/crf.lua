-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'preprocessed.hdf5', 'data file')
cmd:option('-classifier', 'crf', 'classifier to use')
cmd:option('-M',128,'mini-batch size hyperparameter for lr')
cmd:option('-eta',0.01,'learning rate hyperparameter for lr/nn')
cmd:option('-N',5,'num epochs hyperparameter for lr/nn')
cmd:option('-mineta', 0.00001, 'minimum learning rate')
cmd:option('-saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('-load','','specify model to load')
cmd:option('-cuda',false,'whether to use cuda')

function main()
   -- Parse input params
   opt = cmd:parse(arg)
   load()

   if opt.classifier == 'crf' then
     runCRF()
   else
     print("NOT IMPLEMENTED")
   end
end

function runCRF()

end


function precision_recall(y,y_hat,filter)
  filter = filter or 1
  local y_size, y_hat_size = y:size(1),y_hat:size(1)
  local true_positives,found_positives,total_positives = 0,0,0

  for i=1,y_size do
    if y[i] ~= filter then
      total_positives = total_positives + 1
      if i <= y_hat_size and y[i] == y_hat[i] then true_positives = true_positives + 1 end
    end

    if i <= y_hat_size and y_hat[i] ~= filter then
      found_positives = found_positives + 1
    end
  end

  return true_positives,found_positives,total_positives

end

function Fscore(prec,recall,beta)
  beta = beta or 1
  return ((beta^2 + 1)*(prec * recall))/(beta^2 * prec + recall)
end

-- skeleton copied from section
function viterbi(observations, logscore, is_test, sent_number)
  local n = observations:size(1)
  local max_table = torch.Tensor(n, nclasses)
  local backpointer_table = torch.Tensor(n, nclasses)
  local last_step = n

  -- first timestep
  -- the initial most likely paths are the initial state distribution
  -- another unnecessary Tensor allocation here
  --local maxes, backpointers = (initial + emission[observations[1]]):max(2)
  local maxes, backpointers = torch.Tensor(nclasses),torch.Tensor(nclasses)
  maxes:fill(math.log(0))
  maxes[start_tag] = 0
  backpointers:fill(start_tag)

  max_table[1] = maxes

  -- remaining timesteps ("forwarding" the maxes)
  for i=2,n do
    if observations[i] == pad_word then
      last_step = i - 1
      break
    end

    -- precompute edge scores
    y = logscore(observations, i, is_test, sent_number)
    scores = y + maxes:view(1, nclasses):expand(nclasses, nclasses)

    maxes, backpointers = scores:max(2)

    -- record
    max_table[i] = maxes
    backpointer_table[i] = backpointers
  end

  -- follow backpointers to recover max path
  local classes = torch.Tensor(n):fill(pad_tag)

  maxes, classes[last_step] = maxes:max(1)
  for i=last_step,2,-1 do
    classes[i-1] = backpointer_table[{i, classes[i]}]
  end

  return classes
end

function writeToFile(obj,f)
  local myFile = hdf5.open(f, 'w')
  for k,v in pairs(obj) do
    myFile:write(k, v)
  end
  myFile:close()
end

function load()
  local f = hdf5.open(opt.datafile, 'r')
  local data = f:all()

  train_x = d.x_train_with_past
  train_y = d.y_train_with_past

  test_x = d.x_test_with_past
  test_y = d.y_test_with_past
end


main()
