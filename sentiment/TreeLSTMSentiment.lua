--[[

  Sentiment classification using a Binary Tree-LSTM.

--]]

local TreeLSTMSentiment = torch.class('treelstm.TreeLSTMSentiment')

function TreeLSTMSentiment:__init(config)
  self.mem_dim           = config.mem_dim           or 150
  self.learning_rate     = config.learning_rate     or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.batch_size        = config.batch_size        or 25
  self.reg               = config.reg               or 1e-4
  self.structure         = config.structure         or 'constituency'
  self.fine_grained      = (config.fine_grained == nil) and true or config.fine_grained
  self.dropout           = (config.dropout == nil) and true or config.dropout

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  self.in_zeros = torch.zeros(self.emb_dim)
  self.num_classes = self.fine_grained and 5 or 3

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  local treelstm_config = {
    in_dim  = self.emb_dim,
    mem_dim = self.mem_dim,
    output_module_fn = function() return self:new_sentiment_module() end,
    criterion = self.criterion,
  }

  if self.structure == 'dependency' then
    self.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
  elseif self.structure == 'constituency' then
    self.treelstm = treelstm.BinaryTreeLSTM(treelstm_config)
  else
    error('invalid parse tree type: ' .. self.structure)
  end

  self.params, self.grad_params = self.treelstm:getParameters()
end

function TreeLSTMSentiment:new_sentiment_module()
  local sentiment_module = nn.Sequential()
  if self.dropout then
    sentiment_module:add(nn.Dropout())
  end
  sentiment_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())
  return sentiment_module
end

function TreeLSTMSentiment:train(dataset)
  self.treelstm:training()
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local sent = dataset.sents[idx]
        local tree = dataset.trees[idx]

        local inputs = self.emb:forward(sent)
        local _, tree_loss = self.treelstm:forward(tree, inputs)
        loss = loss + tree_loss
        local input_grad = self.treelstm:backward(tree, inputs, {zeros, zeros})
        self.emb:backward(sent, input_grad)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
    self.emb:updateParameters(self.emb_learning_rate)
  end
  xlua.progress(dataset.size, dataset.size)
end

function TreeLSTMSentiment:predict(tree, sent)
  self.treelstm:evaluate()
  local prediction
  local inputs = self.emb:forward(sent)
  self.treelstm:forward(tree, inputs)
  local output = tree.output
  if self.fine_grained then
    prediction = argmax(output)
  else
    prediction = (output[1] > output[3]) and 1 or 3
  end
  self.treelstm:clean(tree)
  return prediction
end

function TreeLSTMSentiment:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(dataset.trees[i], dataset.sents[i])
  end
  return predictions
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function TreeLSTMSentiment:print_config()
  local num_params = self.params:size(1)
  local num_sentiment_params = self:new_sentiment_module():getParameters():size(1)
  printf('%-25s = %s\n',   'fine grained sentiment', tostring(self.fine_grained))
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sentiment_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %s\n',   'dropout', tostring(self.dropout))
end

function TreeLSTMSentiment:save(path)
  local config = {
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs          = self.emb.weight:float(),
    fine_grained      = self.fine_grained,
    learning_rate     = self.learning_rate,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    structure         = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function TreeLSTMSentiment.load(path)
  local state = torch.load(path)
  local model = treelstm.TreeLSTMSentiment.new(state.config)
  model.params:copy(state.params)
  return model
end
