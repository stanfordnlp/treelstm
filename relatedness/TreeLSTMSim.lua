--[[

  Semantic relatedness prediction using Tree-LSTMs.

--]]

local TreeLSTMSim = torch.class('treelstm.TreeLSTMSim')

function TreeLSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.0
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'dependency' -- {dependency, constituency}
  self.sim_nhidden   = config.sim_nhidden   or 50

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  -- number of similarity rating classes
  self.num_classes = 5

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  self.criterion = nn.DistKLDivCriterion()

  -- initialize tree-lstm model
  local treelstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    gate_output = false,
  }
  
  if self.structure == 'dependency' then
    self.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
  elseif self.structure == 'constituency' then
    self.treelstm = treelstm.BinaryTreeLSTM(treelstm_config)
  else
    error('invalid parse tree type: ' .. self.structure)
  end

  -- similarity model
  self.sim_module = self:new_sim_module()
  local modules = nn.Parallel()
    :add(self.treelstm)
    :add(self.sim_module)
  self.params, self.grad_params = modules:getParameters()
end

function TreeLSTMSim:new_sim_module()
  local vecs_to_input
  local lvec = nn.Identity()()
  local rvec = nn.Identity()()
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  vecs_to_input = nn.gModule({lvec, rvec}, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(2 * self.mem_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end

function TreeLSTMSim:train(dataset)
  self.treelstm:training()
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
      self.emb:zeroGradParameters()
      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local ltree, rtree = dataset.ltrees[idx], dataset.rtrees[idx]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        self.emb:forward(lsent)
        local linputs = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
        local rinputs = self.emb:forward(rsent)

        -- get sentence representations
        local lrep = self.treelstm:forward(ltree, linputs)[2]
        local rrep = self.treelstm:forward(rtree, rinputs)[2]

        -- compute relatedness
        local output = self.sim_module:forward{lrep, rrep}

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])
        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, targets[j])
        local rep_grad = self.sim_module:backward({lrep, rrep}, sim_grad)
        local linput_grads = self.treelstm:backward(dataset.ltrees[idx], linputs, {zeros, rep_grad[1]})
        local rinput_grads = self.treelstm:backward(dataset.rtrees[idx], rinputs, {zeros, rep_grad[2]})
        self.emb:backward(lsent, linput_grads)
        self.emb:backward(rsent, rinput_grads)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)
      self.emb:updateParameters(self.emb_learning_rate)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- Predict the similarity of a sentence pair.
function TreeLSTMSim:predict(ltree, rtree, lsent, rsent)
  local linputs = self.emb:forward(lsent)
  local lrep = self.treelstm:forward(ltree, linputs)[2]
  local rinputs = self.emb:forward(rsent)
  local rrep = self.treelstm:forward(rtree, rinputs)[2]
  local output = self.sim_module:forward{lrep, rrep}
  self.treelstm:clean(ltree)
  self.treelstm:clean(rtree)
  return torch.range(1, 5):dot(output:exp())
end

-- Produce similarity predictions for each sentence pair in the dataset.
function TreeLSTMSim:predict_dataset(dataset)
  self.treelstm:evaluate()
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local ltree, rtree = dataset.ltrees[i], dataset.rtrees[i]
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(ltree, rtree, lsent, rsent)
  end
  return predictions
end

function TreeLSTMSim:print_config()
  local num_params = self.params:size(1)
  local num_sim_params = self:new_sim_module():getParameters():size(1)
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %s\n',   'parse tree type', self.structure)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end

--
-- Serialization
--

function TreeLSTMSim:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb.weight:float(),
    learning_rate = self.learning_rate,
    emb_learning_rate = self.emb_learning_rate,
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

function TreeLSTMSim.load(path)
  local state = torch.load(path)
  local model = treelstm.TreeLSTMSim.new(state.config)
  model.params:copy(state.params)
  return model
end
