--[[

  Functions for loading data from disk.

--]]

function treelstm.read_embedding(vocab_path, emb_path)
  local vocab = treelstm.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function treelstm.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

function treelstm.read_trees(parent_path, label_path)
  local parent_file = io.open(parent_path, 'r')
  local label_file
  if label_path ~= nil then label_file = io.open(label_path, 'r') end
  local count = 0
  local trees = {}

  while true do
    local parents = parent_file:read()
    if parents == nil then break end
    parents = stringx.split(parents)
    for i, p in ipairs(parents) do
      parents[i] = tonumber(p)
    end

    local labels
    if label_file ~= nil then
      labels = stringx.split(label_file:read())
      for i, l in ipairs(labels) do
        -- ignore unlabeled nodes
        if l == '#' then
          labels[i] = nil
        else
          labels[i] = tonumber(l)
        end
      end
    end

    count = count + 1
    trees[count] = treelstm.read_tree(parents, labels)
  end
  parent_file:close()
  return trees
end

function treelstm.read_tree(parents, labels)
  local size = #parents
  local trees = {}
  if labels == nil then labels = {} end
  local root
  for i = 1, size do
    if not trees[i] and parents[i] ~= -1 then
      local idx = i
      local prev = nil
      while true do
        local parent = parents[idx]
        if parent == -1 then
          break
        end

        local tree = treelstm.Tree()
        if prev ~= nil then
          tree:add_child(prev)
        end
        trees[idx] = tree
        tree.idx = idx
        tree.gold_label = labels[idx]
        if trees[parent] ~= nil then
          trees[parent]:add_child(tree)
          break
        elseif parent == 0 then
          root = tree
          break
        else
          prev = tree
          idx = parent
        end
      end
    end
  end

  -- index leaves (only meaningful for constituency trees)
  local leaf_idx = 1
  for i = 1, size do
    local tree = trees[i]
    if tree ~= nil and tree.num_children == 0 then
      tree.leaf_idx = leaf_idx
      leaf_idx = leaf_idx + 1
    end
  end
  return root
end

--[[

  Semantic Relatedness

--]]

function treelstm.read_relatedness_dataset(dir, vocab, constituency)
  local dataset = {}
  dataset.vocab = vocab
  if constituency then
    dataset.ltrees = treelstm.read_trees(dir .. 'a.cparents')
    dataset.rtrees = treelstm.read_trees(dir .. 'b.cparents')
  else
    dataset.ltrees = treelstm.read_trees(dir .. 'a.parents')
    dataset.rtrees = treelstm.read_trees(dir .. 'b.parents')
  end
  dataset.lsents = treelstm.read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents = treelstm.read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.ltrees
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readInt()
    dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1)
  end
  id_file:close()
  sim_file:close()
  return dataset
end

--[[

 Sentiment

--]]

function treelstm.read_sentiment_dataset(dir, vocab, fine_grained, dependency)
  local dataset = {}
  dataset.vocab = vocab
  dataset.fine_grained = fine_grained
  local trees
  if dependency then
    trees = treelstm.read_trees(dir .. 'dparents.txt', dir .. 'dlabels.txt')
  else
    trees = treelstm.read_trees(dir .. 'parents.txt', dir .. 'labels.txt')
    for _, tree in ipairs(trees) do
      set_spans(tree)
    end
  end

  local sents = treelstm.read_sentences(dir .. 'sents.txt', vocab)
  if not fine_grained then
    dataset.trees = {}
    dataset.sents = {}
    for i = 1, #trees do
      if trees[i].gold_label ~= 0 then
        table.insert(dataset.trees, trees[i])
        table.insert(dataset.sents, sents[i])
      end
    end
  else
    dataset.trees = trees
    dataset.sents = sents
  end

  dataset.size = #dataset.trees
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    remap_labels(dataset.trees[i], fine_grained)
    dataset.labels[i] = dataset.trees[i].gold_label
  end
  return dataset
end

function set_spans(tree)
  if tree.num_children == 0 then
    tree.lo, tree.hi = tree.leaf_idx, tree.leaf_idx
    return
  end

  for i = 1, tree.num_children do
    set_spans(tree.children[i])
  end

  tree.lo, tree.hi = tree.children[1].lo, tree.children[1].hi
  for i = 2, tree.num_children do
    tree.lo = math.min(tree.lo, tree.children[i].lo)
    tree.hi = math.max(tree.hi, tree.children[i].hi)
  end
end

function remap_labels(tree, fine_grained)
  if tree.gold_label ~= nil then
    if fine_grained then
      tree.gold_label = tree.gold_label + 3
    else
      if tree.gold_label < 0 then
        tree.gold_label = 1
      elseif tree.gold_label == 0 then
        tree.gold_label = 2
      else
        tree.gold_label = 3
      end
    end
  end

  for i = 1, tree.num_children do
    remap_labels(tree.children[i], fine_grained)
  end
end
