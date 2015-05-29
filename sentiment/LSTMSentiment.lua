--[[

  Sentiment classification using LSTMs.

--]]

local LSTMSentiment = torch.class('treelstm.LSTMSentiment')

function LSTMSentiment:__init(config)
  self.mem_dim           = config.mem_dim           or 150
  self.learning_rate     = config.learning_rate     or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.num_layers        = config.num_layers        or 1
  self.batch_size        = config.batch_size        or 5
  self.reg               = config.reg               or 1e-4
  self.structure         = config.structure         or 'lstm' -- {lstm, bilstm}
  self.fine_grained      = (config.fine_grained == nil) and true or config.fine_grained
  self.dropout           = (config.dropout == nil) and true or config.dropout
  self.train_subtrees    = 4  -- number of subtrees to sample during training

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

  -- sentiment classification module
  self.sentiment_module = self:new_sentiment_module()

  -- initialize LSTM model
  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = true,
  }

  if self.structure == 'lstm' then
    self.lstm = treelstm.LSTM(lstm_config)
  elseif self.structure == 'bilstm' then
    self.lstm = treelstm.LSTM(lstm_config)
    self.lstm_b = treelstm.LSTM(lstm_config)
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  local modules = nn.Parallel()
    :add(self.lstm)
    :add(self.sentiment_module)
  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  if self.structure == 'bilstm' then
    share_params(self.lstm_b, self.lstm)
  end
end

function LSTMSentiment:new_sentiment_module()
  local input_dim = self.num_layers * self.mem_dim
  local inputs, vec
  if self.structure == 'lstm' then
    local rep = nn.Identity()()
    if self.num_layers == 1 then
      vec = {rep}
    else
      vec = nn.JoinTable(1)(rep)
    end
    inputs = {rep}
  elseif self.structure == 'bilstm' then
    local frep, brep = nn.Identity()(), nn.Identity()()
    input_dim = input_dim * 2
    if self.num_layers == 1 then
      vec = nn.JoinTable(1){frep, brep}
    else
      vec = nn.JoinTable(1){nn.JoinTable(1)(frep), nn.JoinTable(1)(brep)}
    end
    inputs = {frep, brep}
  end

  local logprobs
  if self.dropout then
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(
        nn.Dropout()(vec)))
  else
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(vec))
  end

  return nn.gModule(inputs, {logprobs})
end

function LSTMSentiment:train(dataset)
  self.lstm:training()
  self.sentiment_module:training()
  if self.structure == 'bilstm' then
    self.lstm_b:training()
  end

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
        local tree = dataset.trees[idx]
        local sent = dataset.sents[idx]
        local subtrees = tree:depth_first_preorder()
        for k = 1, self.train_subtrees + 1 do
          local subtree = (k == 1) and tree or subtrees[math.ceil(torch.uniform(1, #subtrees))]
          local span = sent[{{subtree.lo, subtree.hi}}]
          local inputs = self.emb:forward(span)

          -- get sentence representations
          local rep
          if self.structure == 'lstm' then
            rep = self.lstm:forward(inputs)
          elseif self.structure == 'bilstm' then
            rep = {
              self.lstm:forward(inputs),
              self.lstm_b:forward(inputs, true), -- true => reverse
            }
          end

          -- compute class log probabilities
          local output = self.sentiment_module:forward(rep)

          -- compute loss and backpropagate
          local example_loss = self.criterion:forward(output, subtree.gold_label)
          loss = loss + example_loss
          local obj_grad = self.criterion:backward(output, subtree.gold_label)
          local rep_grad = self.sentiment_module:backward(rep, obj_grad)
          local input_grads
          if self.structure == 'lstm' then
            input_grads = self:LSTM_backward(sent, inputs, rep_grad)
          elseif self.structure == 'bilstm' then
            input_grads = self:BiLSTM_backward(sent, inputs, rep_grad)
          end
          self.emb:backward(span, input_grads)
        end
      end

      local batch_subtrees = batch_size * (self.train_subtrees + 1)
      loss = loss / batch_subtrees
      self.grad_params:div(batch_subtrees)
      self.emb.gradWeight:div(batch_subtrees)

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

-- LSTM backward propagation
function LSTMSentiment:LSTM_backward(sent, inputs, rep_grad)
  local grad
  if self.num_layers == 1 then
    grad = torch.zeros(sent:nElement(), self.mem_dim)
    grad[sent:nElement()] = rep_grad
  else
    grad = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      grad[{sent:nElement(), l, {}}] = rep_grad[l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  return input_grads
end

-- Bidirectional LSTM backward propagation
function LSTMSentiment:BiLSTM_backward(sent, inputs, rep_grad)
  local grad, grad_b
  if self.num_layers == 1 then
    grad   = torch.zeros(sent:nElement(), self.mem_dim)
    grad_b = torch.zeros(sent:nElement(), self.mem_dim)
    grad[sent:nElement()] = rep_grad[1]
    grad_b[1] = rep_grad[2]
  else
    grad   = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    grad_b = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      grad[{sent:nElement(), l, {}}] = rep_grad[1][l]
      grad_b[{1, l, {}}] = rep_grad[2][l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  local input_grads_b = self.lstm_b:backward(inputs, grad_b, true)
  return input_grads + input_grads_b
end

-- Predict the sentiment of a sentence.
function LSTMSentiment:predict(sent)
  self.lstm:evaluate()
  self.sentiment_module:evaluate()
  local inputs = self.emb:forward(sent)

  local rep
  if self.structure == 'lstm' then
    rep = self.lstm:forward(inputs)
  elseif self.structure == 'bilstm' then
    self.lstm_b:evaluate()
    rep = {
      self.lstm:forward(inputs),
      self.lstm_b:forward(inputs, true),
    }
  end
  local logprobs = self.sentiment_module:forward(rep)
  local prediction
  if self.fine_grained then
    prediction = argmax(logprobs)
  else
    prediction = (logprobs[1] > logprobs[3]) and 1 or 3
  end
  self.lstm:forget()
  if self.structure == 'bilstm' then
    self.lstm_b:forget()
  end
  return prediction
end

-- Produce sentiment predictions for each sentence in the dataset.
function LSTMSentiment:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(dataset.sents[i])
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

function LSTMSentiment:print_config()
  local num_params = self.params:size(1)
  local num_sentiment_params = self:new_sentiment_module():getParameters():size(1)
  printf('%-25s = %s\n',   'fine grained sentiment', tostring(self.fine_grained))
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sentiment_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
  printf('%-25s = %s\n',   'LSTM structure', self.structure)
  printf('%-25s = %d\n',   'LSTM layers', self.num_layers)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size * (self.train_subtrees + 1))
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %s\n',   'dropout', tostring(self.dropout))
end

--
-- Serialization
--

function LSTMSentiment:save(path)
  local config = {
    batch_size        = self.batch_size,
    dropout           = self.dropout,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs          = self.emb.weight:float(),
    fine_grained      = self.fine_grained,
    learning_rate     = self.learning_rate,
    num_layers        = self.num_layers,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    structure         = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function LSTMSentiment.load(path)
  local state = torch.load(path)
  local model = treelstm.LSTMSentiment.new(state.config)
  model.params:copy(state.params)
  return model
end
