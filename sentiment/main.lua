--[[

  Tree-LSTM training script for sentiment classication on the Stanford
  Sentiment Treebank

--]]

require('..')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

header('Tree-LSTM for Sentiment Classification')

-- binary or fine-grained subtask
local fine_grained = true
if arg[1] == '--binary' then
  fine_grained = false
end

-- directory containing dataset files
local data_dir = 'data/sst/'

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
  local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
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
local train_dataset = treelstm.read_sentiment_dataset(train_dir, vocab, fine_grained)
local dev_dataset = treelstm.read_sentiment_dataset(dev_dir, vocab, fine_grained)
local test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, fine_grained)

printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local model = treelstm.TreeLSTMSentiment{
  emb_vecs = vecs,
  fine_grained = fine_grained,
}

-- number of epochs to train
local num_epochs = 10

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
header('Training Tree-LSTM')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train(train_dataset)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)
  -- local train_predictions = model:predict_dataset(train_dataset)
  -- local train_score = accuracy(train_predictions, train_dataset.labels)
  -- printf('-- train score: %.4f\n', train_score)
  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_score = accuracy(dev_predictions, dev_dataset.labels)
  printf('-- dev score: %.4f\n', dev_score)

  if dev_score > best_dev_score then
    best_dev_score = dev_score
    best_dev_model = treelstm.TreeLSTMSentiment{
      emb_vecs = vecs,
      fine_grained = fine_grained,
    }
    best_dev_model.params:copy(model.params)
    best_dev_model.emb.weight:copy(model.emb.weight)
  end
end
printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)
printf('-- test score: %.4f\n', accuracy(test_predictions, test_dataset.labels))

-- write predictions to disk
local subtask = fine_grained and '5class' or '2class'
if lfs.attributes(treelstm.predictions_dir) == nil then
  lfs.mkdir(treelstm.predictions_dir)
end
local predictions_save_path = string.format(
  treelstm.predictions_dir .. '/sent-treelstm.%d.%s.pred', model.mem_dim, subtask)
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing predictions to ' .. predictions_save_path)
for i = 1, test_predictions:size(1) do
  predictions_file:writeInt(test_predictions[i])
end
predictions_file:close()

-- write model to disk
if lfs.attributes(treelstm.models_dir) == nil then
  lfs.mkdir(treelstm.models_dir)
end
local model_save_path = string.format(
  treelstm.models_dir .. '/sent-treelstm.%d.%s.th', model.mem_dim, subtask)
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

-- to load a saved model
-- local loaded = treelstm.TreeLSTMSentiment.load(model_save_path)
