using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = ENV["USE_GPU"] == "true" ? true : false

@static if USE_GPU
  @init_parallel_stencil(CUDA, Float64, 2)
else
  @init_parallel_stencil(Threads, Float64, 2)
end

include("../scripts/multi_process/MarkerExchange.jl")

using Test, ImplicitGlobalGrid
import MPI

MPI.Init()
@assert MPI.Comm_size(MPI.COMM_WORLD) == 9 "This test needs 9 processes"
rank, dims, nprocs, coords, comm_cart = init_global_grid(3, 3, 1; dimx=3, dimy=3, init_MPI=false, quiet=true)

# init markers in all eight directions and a single one in the centre
localDomain = [1.0, 1.0]
dx, dy = 0.4, 0.4 # => 0.8 x 0.8 local domain except at physical boundaries
x_m = Data.Array([0.1 + 0.4 * (i % 3) for i = 0:8])
y_m = Data.Array([0.1 + 0.4 * (i ÷ 3) for i = 0:8])
ρ_m = Data.Array([coords[1] for _ = 0:8]) # encode source coords for testing later
μ_m = Data.Array([coords[2] for _ = 0:8])

# the centre rank now has markers at the following positions relative to the local domain indicated by the box:
# x       x       x
#     ---------    
#     |       |    
# x   |   x   |   x
#     |       |    
#     ---------    
# x       x       x

# the bottom left rank:
#   x       x       x
# -----------------
# |               | 
# | x       x     | x
# |               | 
# |               | 
# | x       x     | x
# -----------------

# ... and so on.

# store number of markers before exchange
sz = size(x_m, 1)
@assert sz == size(y_m, 1)
@assert sz == size(ρ_m, 1)
@assert sz == size(μ_m, 1)

# exchange markers
x_m, y_m, ρ_m, μ_m = exchangeMarkers(comm_cart, dims, localDomain, dx, dy, Array(x_m), Array(y_m), Array(ρ_m), Array(μ_m))

# check whether no markers have been lost (number of sent markers equals the number of received markers here)
@test sz == size(x_m, 1)
@test sz == size(y_m, 1)
@test sz == size(ρ_m, 1)
@test sz == size(μ_m, 1)

# check all marker coords and all src ranks for the centre rank
if coords[1] == 1 && coords[2] == 1
  # sort by coordinates
  m = sortslices([x_m y_m ρ_m μ_m]', dims=2)

  # bottom left marker
  @test m[1] ≈ 0.3 # x
  @test m[2] ≈ 0.3 # y
  @test m[3] == 0 && m[4] == 0 # bottom left src rank
  # centre left marker
  @test m[5] ≈ 0.3 # x
  @test m[6] ≈ 0.5 # y
  @test m[7] == 0 && m[8] == 1 # centre left src rank
  # top left marker
  @test m[9] ≈ 0.3 # x
  @test m[10] ≈ 0.7 # y
  @test m[11] == 0 && m[12] == 2 # top left src rank
  # bottom centre marker
  @test m[13] ≈ 0.5 # x
  @test m[14] ≈ 0.3 # y
  @test m[15] == 1 && m[16] == 0 # bottom centre src rank
  # centre marker
  @test m[17] ≈ 0.5 # x
  @test m[18] ≈ 0.5 # y
  @test m[19] == 1 && m[20] == 1 # centre src rank
  # top centre marker
  @test m[21] ≈ 0.5 # x
  @test m[22] ≈ 0.7 # y
  @test m[23] == 1 && m[24] == 2 # top centre src rank
  # bottom right marker
  @test m[25] ≈ 0.7 # x
  @test m[26] ≈ 0.3 # y
  @test m[27] == 2 && m[28] == 0 # bottom right src rank
  # centre right marker
  @test m[29] ≈ 0.7 # x
  @test m[30] ≈ 0.5 # y
  @test m[31] == 2 && m[32] == 1 # centre right src rank
  # top right marker
  @test m[33] ≈ 0.7 # x
  @test m[34] ≈ 0.7 # y
  @test m[35] == 2 && m[36] == 2 # top right src rank
end

finalize_global_grid()
