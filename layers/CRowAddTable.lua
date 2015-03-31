--[[

  Add a vector to every row of a matrix.

  Input: { [n x m], [m] }

  Output: [n x m]

--]]

local CRowAddTable, parent = torch.class('treelstm.CRowAddTable', 'nn.Module')

function CRowAddTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CRowAddTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   for i = 1, self.output:size(1) do
      self.output[i]:add(input[2])
   end
   return self.output
end

function CRowAddTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.gradInput[1]:resizeAs(input[1])
   self.gradInput[2]:resizeAs(input[2]):zero()

   self.gradInput[1]:copy(gradOutput)
   for i = 1, gradOutput:size(1) do
      self.gradInput[2]:add(gradOutput[i])
   end

   return self.gradInput
end
