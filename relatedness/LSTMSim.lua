--[[

  Semantic relatedness prediction using LSTMs.

--]]

local LSTMSim = torch.class('treelstm.LSTMSim')

function LSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
  self.sim_nhidden   = config.sim_nhidden   or 50

  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)

  -- number of similarity rating classes
  self.num_classes = 5

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  self.criterion = nn.DistKLDivCriterion()

  -- initialize LSTM model
  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
    gate_output = false,
  }

  if self.structure == 'lstm' then
    self.llstm = treelstm.LSTM(lstm_config) -- "left" LSTM
    self.rlstm = treelstm.LSTM(lstm_config) -- "right" LSTM
  elseif self.structure == 'bilstm' then
    self.llstm = treelstm.LSTM(lstm_config)
    self.llstm_b = treelstm.LSTM(lstm_config) -- backward "left" LSTM
    self.rlstm = treelstm.LSTM(lstm_config)
    self.rlstm_b = treelstm.LSTM(lstm_config) -- backward "right" LSTM
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  -- similarity model
  self.sim_module = self:new_sim_module()
  local modules = nn.Parallel()
    :add(self.llstm)
    :add(self.sim_module)
  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  share_params(self.rlstm, self.llstm)
  if self.structure == 'bilstm' then
    -- tying the forward and backward weights improves performance
    share_params(self.llstm_b, self.llstm)
    share_params(self.rlstm_b, self.llstm)
  end
end

function LSTMSim:new_sim_module()
  local lvec, rvec, inputs, input_dim
  if self.structure == 'lstm' then
    -- standard (left-to-right) LSTM
    input_dim = 2 * self.num_layers * self.mem_dim
    local linput, rinput = nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec, rvec = linput, rinput
    else
      lvec, rvec = nn.JoinTable(1)(linput), nn.JoinTable(1)(rinput)
    end
    inputs = {linput, rinput}
  elseif self.structure == 'bilstm' then
    -- bidirectional LSTM
    input_dim = 4 * self.num_layers * self.mem_dim
    local lf, lb, rf, rb = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec = nn.JoinTable(1){lf, lb}
      rvec = nn.JoinTable(1){rf, rb}
    else
      -- in the multilayer case, each input is a table of hidden vectors (one for each layer)
      lvec = nn.JoinTable(1){nn.JoinTable(1)(lf), nn.JoinTable(1)(lb)}
      rvec = nn.JoinTable(1){nn.JoinTable(1)(rf), nn.JoinTable(1)(rb)}
    end
    inputs = {lf, lb, rf, rb}
  end
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  local vecs_to_input = nn.gModule(inputs, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(input_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end

function LSTMSim:train(dataset)
  self.llstm:training()
  self.rlstm:training()
  if self.structure == 'bilstm' then
    self.llstm_b:training()
    self.rlstm_b:training()
  end

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    -- get target distributions for batch
    local targets = torch.zeros(batch_size, self.num_classes)
    for j = 1, batch_size do
      local sim = dataset.labels[indices[i + j - 1]] * (self.num_classes - 1) + 1
      local ceil, floor = math.ceil(sim), math.floor(sim)
      if ceil == floor then
        targets[{j, floor}] = 1
      else
        targets[{j, floor}] = ceil - sim
        targets[{j, ceil}] = sim - floor
      end
    end

    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local linputs = self.emb_vecs:index(1, lsent:long()):double()
        local rinputs = self.emb_vecs:index(1, rsent:long()):double()

        -- get sentence representations
        local inputs
        if self.structure == 'lstm' then
          inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
        elseif self.structure == 'bilstm' then
          inputs = {
            self.llstm:forward(linputs),
            self.llstm_b:forward(linputs, true), -- true => reverse
            self.rlstm:forward(rinputs),
            self.rlstm_b:forward(rinputs, true)
          }
        end

        -- compute relatedness
        local output = self.sim_module:forward(inputs)

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])
        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        local rep_grad = self.sim_module:backward(inputs, sim_grad)
        if self.structure == 'lstm' then
          self:LSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
        elseif self.structure == 'bilstm' then
          self:BiLSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
        end
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function LSTMSim:LSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
  local lgrad, rgrad
  if self.num_layers == 1 then
    lgrad = torch.zeros(lsent:nElement(), self.mem_dim)
    rgrad = torch.zeros(rsent:nElement(), self.mem_dim)
    lgrad[lsent:nElement()] = rep_grad[1]
    rgrad[rsent:nElement()] = rep_grad[2]
  else
    lgrad = torch.zeros(lsent:nElement(), self.num_layers, self.mem_dim)
    rgrad = torch.zeros(rsent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      lgrad[{lsent:nElement(), l, {}}] = rep_grad[1][l]
      rgrad[{rsent:nElement(), l, {}}] = rep_grad[2][l]
    end
  end
  self.llstm:backward(linputs, lgrad)
  self.rlstm:backward(rinputs, rgrad)
end

-- Bidirectional LSTM backward propagation
function LSTMSim:BiLSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
  local lgrad, lgrad_b, rgrad, rgrad_b
  if self.num_layers == 1 then
    lgrad   = torch.zeros(lsent:nElement(), self.mem_dim)
    lgrad_b = torch.zeros(lsent:nElement(), self.mem_dim)
    rgrad   = torch.zeros(rsent:nElement(), self.mem_dim)
    rgrad_b = torch.zeros(rsent:nElement(), self.mem_dim)
    lgrad[lsent:nElement()] = rep_grad[1]
    rgrad[rsent:nElement()] = rep_grad[3]
    lgrad_b[1] = rep_grad[2]
    rgrad_b[1] = rep_grad[4]
  else
    lgrad   = torch.zeros(lsent:nElement(), self.num_layers, self.mem_dim)
    lgrad_b = torch.zeros(lsent:nElement(), self.num_layers, self.mem_dim)
    rgrad   = torch.zeros(rsent:nElement(), self.num_layers, self.mem_dim)
    rgrad_b = torch.zeros(rsent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      lgrad[{lsent:nElement(), l, {}}] = rep_grad[1][l]
      rgrad[{rsent:nElement(), l, {}}] = rep_grad[3][l]
      lgrad_b[{1, l, {}}] = rep_grad[2][l]
      rgrad_b[{1, l, {}}] = rep_grad[4][l]
    end
  end
  self.llstm:backward(linputs, lgrad)
  self.llstm_b:backward(linputs, lgrad_b, true)
  self.rlstm:backward(rinputs, rgrad)
  self.rlstm_b:backward(rinputs, rgrad_b, true)
end

-- Predict the similarity of a sentence pair.
function LSTMSim:predict(lsent, rsent)
  self.llstm:evaluate()
  self.rlstm:evaluate()
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()
  local inputs
  if self.structure == 'lstm' then
    inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
  elseif self.structure == 'bilstm' then
    self.llstm_b:evaluate()
    self.rlstm_b:evaluate()
    inputs = {
      self.llstm:forward(linputs),
      self.llstm_b:forward(linputs, true),
      self.rlstm:forward(rinputs),
      self.rlstm_b:forward(rinputs, true)
    }
  end
  local output = self.sim_module:forward(inputs)
  self.llstm:forget()
  self.rlstm:forget()
  if self.structure == 'bilstm' then
    self.llstm_b:forget()
    self.rlstm_b:forget()
  end
  return torch.range(1, 5):dot(output:exp())
end

-- Produce similarity predictions for each sentence pair in the dataset.
function LSTMSim:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(lsent, rsent)
  end
  return predictions
end

function LSTMSim:print_config()
  local num_params = self.params:nElement()
  local num_sim_params = self:new_sim_module():getParameters():nElement()
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %s\n',   'LSTM structure', self.structure)
  printf('%-25s = %d\n',   'LSTM layers', self.num_layers)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end

--
-- Serialization
--

function LSTMSim:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb_vecs:float(),
    learning_rate = self.learning_rate,
    num_layers    = self.num_layers,
    mem_dim       = self.mem_dim,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function LSTMSim.load(path)
  local state = torch.load(path)
  local model = treelstm.LSTMSim.new(state.config)
  model.params:copy(state.params)
  return model
end
