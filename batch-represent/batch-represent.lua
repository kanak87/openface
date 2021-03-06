local ffi = require 'ffi'

local batchNumber, nImgs = 0

function batchRepresent()
   local loadSize   = {3, opt.imgDim, opt.imgDim}
   local dumpLoader = dataLoader{
      paths = {opt.data},
      loadSize = loadSize,
      sampleSize = loadSize,
      split = 0,
      verbose = true
   }
   nImgs = dumpLoader:sizeTest()
   print('nImgs: ', nImgs)
   assert(nImgs > 0, "Failed to get nImgs")

   batchNumber = 0

   for i=1,math.ceil(nImgs/opt.batchSize) do
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = math.min(nImgs, indexStart + opt.batchSize - 1)
      local inputs, labels = dumpLoader:get(indexStart, indexEnd)
      local paths = {}
      for i=indexStart,indexEnd do
         table.insert(paths, ffi.string(dumpLoader.imagePath[i]:data()))
      end
      repBatch(paths, inputs, labels)
      if i % 5 == 0 then
         collectgarbage()
      end
   end

   if opt.cuda then
      cutorch.synchronize()
   end
end

function repBatch(paths, inputs, labels)
   -- labels:size(1) is equal to batchSize except for the last iteration if
   -- the number of images isn't equal to the batch size.
   local n = labels:size(1)
   batchNumber = batchNumber + n

   if opt.cuda then
      inputs = inputs:cuda()
   end
   local embeddings = model:forward(inputs):float()
   if opt.cuda then
      cutorch.synchronize()
   end

   for i=1,n do
      labelsCSV:write({labels[i], paths[i]})
      repsCSV:write(embeddings[i]:totable())
   end

   print(('Represent: %d/%d'):format(batchNumber, nImgs))
end
