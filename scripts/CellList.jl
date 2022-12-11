using Test, CUDA

mutable struct CellList
  # inverse cell sizes for each dimension
  invCellSize::CuArray{Float32}

  # number of cells for each dimension
  numCells::CuArray{Int32}

  # cell list arrays
  counts::CuArray{Int32}
  offsets::CuArray{Int32}

  CellList() = new()
end

@views function CellList_init(domainSize_x, domainSize_y, cellSize)
  @assert domainSize_x > 0.0
  @assert domainSize_y > 0.0
  @assert cellSize > 0.0

  cl = CellList()

  numCells_x = ceil(Int32, domainSize_x / cellSize)
  numCells_y = ceil(Int32, domainSize_y / cellSize)
  cl.numCells = CuArray([numCells_x, numCells_y])

  invCellSize_x = numCells_x / domainSize_x
  invCellSize_y = numCells_y / domainSize_y
  cl.invCellSize = CuArray([invCellSize_x, invCellSize_y])

  totalCells = numCells_x * numCells_y
  cl.counts = CUDA.zeros(totalCells)
  cl.offsets = CUDA.zeros(totalCells)

  return cl
end

function CellList_getCell!(invCellSize, p1, p2)
  return trunc(Int32, p1 * invCellSize[1]), trunc(Int32, p2 * invCellSize[2])
end

function CellList_getCellIndex!(invCellSize, numCells, p1, p2)
  c1, c2 = CellList_getCell!(invCellSize, p1, p2)
  return c2 * numCells[1] + c1
end

function CellList_inclusiveSum!(cl)
  # array programming, needs no kernel launch
  cumsum!(cl.offsets, cl.counts; dims=1)
  return nothing
end

function CellList_countParticlePerCell!(invCellSize, numCells, counts, p, numParticles)
  i = blockIdx().x * blockDim().x + threadIdx().x
  while i <= numParticles
    cIdx = CellList_getCellIndex!(invCellSize, numCells, p[1, i], p[2, i])
    #CUDA.atomic_add!(pointer(counts, cIdx), 1)
    CUDA.@atomic counts[cIdx] += one(eltype(counts))
    i += gridDim().x * blockDim().x
  end
  return nothing
end

function CellList_rearrangeParticles!(invCellSize, numCells, counts, offsets, p, pNew, numParticles)
  i = blockIdx().x * blockDim().x + threadIdx().x
  while i <= numParticles
    cIdx = CellList_getCellIndex!(invCellSize, numCells, p[1, i], p[2, i])
    #newIndex = offsets[cIdx] + atomic_add!(pointer(counts, cIdx), 1)
    newIndex = offsets[cIdx] + counts[cIdx]
    CUDA.@atomic counts[cIdx] += one(eltype(counts))
    pNew[1, newIndex] = p[1, i]
    pNew[2, newIndex] = p[2, i]
    i += gridDim().x * blockDim().x
  end
  return nothing
end

@views function CellList_build(cl::CellList, pDev, pSortedDev, numParticles)
  #threads = 256
  #blocks = (numParticles + threads - 1) / threads
  # CUDA.@sync @cuda blocks = blocks threads = threads CellList_countParticlePerCell!(...)

  # stage 1: compute cell sizes
  cl.counts .= 0
  CUDA.@sync @cuda CellList_countParticlePerCell!(cl.invCellSize, cl.numCells, cl.counts, pDev, numParticles)

  # stage 2: compute offsets
  CellList_inclusiveSum!(cl)

  # stage 3: reorder particles into cells
  cl.counts .= 0
  CUDA.@sync @cuda CellList_rearrangeParticles!(cl.invCellSize, cl.numCells, cl.counts, cl.offsets, pDev, pSortedDev, numParticles)
end

@testset "Cell List" begin
  #GC.gc()
  #GC.enable(false)

  domainSize_x = 1.0
  domainSize_y = 1.0
  cellSize = 0.1
  cl = CellList_init(domainSize_x, domainSize_y, cellSize)

  numCells = Array(cl.numCells)
  @test size(numCells, 1) == 2
  @test numCells[1] == 10
  @test numCells[2] == 10

  invCellSize = Array(cl.invCellSize)
  @test size(invCellSize, 1) == 2
  @test invCellSize[1] ≈ 10.0
  @test invCellSize[2] ≈ 10.0

  @test size(cl.counts, 1) == 100
  @test size(cl.offsets, 1) == 100

  numParticles = 10
  pHost = zeros(2, numParticles)
  for i = 1:numParticles
    pHost[1, i] = rand() * domainSize_x
    pHost[2, i] = rand() * domainSize_y
  end
  pDev = CuArray(pHost)
  pSortedDev = CuArray(pHost)

  @show Array(pDev)
  CellList_build(cl, pDev, pSortedDev, numParticles)
  @show Array(pSortedDev)

  #GC.enable(true)
end
