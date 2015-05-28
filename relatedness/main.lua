--[[

  Training script for semantic relatedness prediction on the SICK dataset.

--]]

require('..')

-- Pearson correlation
function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

-- read command line arguments
local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -m,--model  (default dependency) Model architecture: [dependency, constituency, lstm, bilstm]
  -l,--layers (default 1)          Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)        LSTM memory dimension
  -e,--epochs (default 10)         Number of training epochs
]]

local model_name, model_class
if args.model == 'dependency' then
  model_name = 'Dependency Tree LSTM'
  model_class = treelstm.TreeLSTMSim
elseif args.model == 'constituency' then
  model_name = 'Constituency Tree LSTM'
  model_class = treelstm.TreeLSTMSim
elseif args.model == 'lstm' then
  model_name = 'LSTM'
  model_class = treelstm.LSTMSim
elseif args.model == 'bilstm' then
  model_name = 'Bidirectional LSTM'
  model_class = treelstm.LSTMSim
end
local model_structure = args.model
header(model_name .. ' for Semantic Relatedness')

-- directory containing dataset files
local data_dir = 'data/sick/'

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local constituency = (args.model == 'constituency')
local train_dataset = treelstm.read_relatedness_dataset(train_dir, vocab, constituency)
local dev_dataset = treelstm.read_relatedness_dataset(dev_dir, vocab, constituency)
local test_dataset = treelstm.read_relatedness_dataset(test_dir, vocab, constituency)
printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local model = model_class{
  emb_vecs   = vecs,
  structure  = model_structure,
  num_layers = args.layers,
  mem_dim    = args.dim,
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train(train_dataset)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  -- uncomment to compute train scores
  --[[
  local train_predictions = model:predict_dataset(train_dataset)
  local train_score = pearson(train_predictions, train_dataset.labels)
  printf('-- train score: %.4f\n', train_score)
  --]]

  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_score = pearson(dev_predictions, dev_dataset.labels)
  printf('-- dev score: %.4f\n', dev_score)

  if dev_score > best_dev_score then
    best_dev_score = dev_score
    best_dev_model = model_class{
      emb_vecs = vecs,
      structure = model_structure,
      num_layers = args.layers,
      mem_dim    = args.dim,
    }
    best_dev_model.params:copy(model.params)
  end
end
printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)
local test_score = pearson(test_predictions, test_dataset.labels)
printf('-- test score: %.4f\n', test_score)

-- create predictions and model directories if necessary
if lfs.attributes(treelstm.predictions_dir) == nil then
  lfs.mkdir(treelstm.predictions_dir)
end

if lfs.attributes(treelstm.models_dir) == nil then
  lfs.mkdir(treelstm.models_dir)
end

-- get paths
local file_idx = 1
local predictions_save_path, model_save_path
while true do
  predictions_save_path = string.format(
    treelstm.predictions_dir .. '/rel-%s.%dl.%dd.%d.pred', args.model, args.layers, args.dim, file_idx)
  model_save_path = string.format(
    treelstm.models_dir .. '/rel-%s.%dl.%dd.%d.th', args.model, args.layers, args.dim, file_idx)
  if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

-- write predictions to disk
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing predictions to ' .. predictions_save_path)
for i = 1, test_predictions:size(1) do
  predictions_file:writeFloat(test_predictions[i])
end
predictions_file:close()

-- write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

-- to load a saved model
-- local loaded = model_class.load(model_save_path)
