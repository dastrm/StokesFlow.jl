using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = ENV["USE_GPU"] == "true" ? true : false

@static if USE_GPU
  @init_parallel_stencil(CUDA, Float64, 2)
else
  @init_parallel_stencil(Threads, Float64, 2)
end

include("../scripts/multi_process/MarkerExchange.jl")
include("../scripts/multi_process/common.jl")

using Test, ImplicitGlobalGrid
import MPI, Random

MPI.Init()
@assert MPI.Comm_size(MPI.COMM_WORLD) == 1 "This test needs 1 process"
Nx, Ny = 7, 11
rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1; init_MPI=false, quiet=true)

# init parameters and arrays
lx, ly = 17, 13
dx = lx / (Nx - 1)
dy = ly / (Ny - 1)
marker_density = 3
Nmx = (Nx - 1) * marker_density
Nmy = (Ny - 1) * marker_density
Nm = Nmx * Nmy
x = [(ix - 1) * dx for ix = 1:Nx]
y = [(iy - 1) * dy for iy = 1:Ny]
localDomain = [lx / dims[1], ly / dims[2]]

# init markers with rng
x_m = zeros(Nm)
y_m = zeros(Nm)
Random.seed!(rank)
setInitialMarkerCoords!(coords, dims, dx, dy, x_m, y_m, Nmx, Nmy, x, y, true)
ρ_m = rand(Nm)
μ_m = rand(Nm)

# exchange markers
x_me, y_me, ρ_me, μ_me = exchangeMarkers!(comm_cart, dims, localDomain, dx, dy, Array(x_m), Array(y_m), Array(ρ_m), Array(μ_m))

# check whether markers have been altered in any way
# (this also implicily tests whether setInitialMarkerCoords! sets no markers outside of the local boundary, because exchangeMarkers! removes those)
@test all(x_m .≈ x_me)
@test all(y_m .≈ y_me)
@test all(ρ_m .≈ ρ_me)
@test all(μ_m .≈ μ_me)

finalize_global_grid()
