using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = ENV["USE_GPU"] == "true" ? true : false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

include("../scripts/multi_process/MarkerToGrid.jl")

using Test
import MPI

MPI.Init()

size = MPI.Comm_size(MPI.COMM_WORLD)
@test size == 4

MPI.Finalize()

