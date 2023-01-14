using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = ENV["USE_GPU"] == "true" ? true : false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

include("../scripts/multi_process/MarkerToGrid.jl")
include("../scripts/GlobalGather.jl")

using Test, ImplicitGlobalGrid
import MPI

Nx = 5
Ny = 4
rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1; init_MPI=true, quiet=true)
grid = ImplicitGlobalGrid.get_global_grid()

@assert MPI.Comm_size(comm_cart) == 4 "This test needs 4 processes"

# TEST 1: no overlap, 3x2 array, one marker per cell
x_grid_min = 3.0;
dx = 1.4;
y_grid_min = 4.5;
dy = 2.0;
x_m = Data.Array([4.3, 5.6])
y_m = Data.Array([6.0, 4.8])
val_m = Data.Array([13.5, 10.0])
val_grid = @zeros(3, 2)
val_wt_sum = @ones(3, 2)
wt_sum = @ones(3, 2)
bilinearMarkerToGrid!(x_grid_min, y_grid_min, val_grid, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum, grid)
reference = [[13.5, 12.297979797979796, 10.0] [13.5, 13.395522388059701, 10.0]]; # manually computed
@test all(Array(val_grid) .≈ reference)
finalize_global_grid(; finalize_MPI=false)


# --------------------------------------------------------------------------------------------------------------

# TEST 2: compare single and multi-process (2x2) results

# 1) -------- INITIALIZE AND COMPUTE WITH 4 processes --------
Nx = 10
Ny = 8
rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1; init_MPI=false, dimx=2, dimy=2, dimz=1, quiet=true)
grid = ImplicitGlobalGrid.get_global_grid()
# global grid is 18x14
x_grid_min = 0.0;
dx = 0.3;
y_grid_min = 0.0;
dy = 0.4;
lx = (Nx - 1) * dx;
ly = (Ny - 1) * dy;

# grids
val_grid1 = @zeros(Nx - 1, Ny - 1) # one/one overlap
val_grid2 = @zeros(Nx, Ny)         # two/two overlap
val_grid3 = @zeros(Nx + 1, Ny)     # three/two overlap
x_grid1_min = 0.5dx
x_grid2_min = 0.0
x_grid3_min = -0.5dx
y_grid1_min = 0.5dy
y_grid2_min = 0.0
y_grid3_min = 0.0

# helper arrays, just as large as needed
val_wt_sum = @ones(Nx + 1, Ny)
wt_sum = @ones(Nx + 1, Ny)

# marker coords
Nm = 25 * (Nx - 1) * (Ny - 1)
x_m = @rand(Nm) .* (lx - 0.5dx) .+ (coords[1] > 0 ? 0.5dx : 0.0)
y_m = @rand(Nm) .* (ly - 0.5dy) .+ (coords[2] > 0 ? 0.5dy : 0.0)
val_m = @rand(Nm) .+ 2 * (rank + 1)

# interpolate to grid
bilinearMarkerToGrid!(x_grid1_min, y_grid1_min, val_grid1, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum, grid)
bilinearMarkerToGrid!(x_grid2_min, y_grid2_min, val_grid2, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum, grid)
bilinearMarkerToGrid!(x_grid3_min, y_grid3_min, val_grid3, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum, grid)

# gather global marker values
x_m_glob = zeros(0)
y_m_glob = zeros(0)
val_m_glob = zeros(0)
gather_markers!(Array(x_m), Array(y_m), Array(val_m), Array(val_m), x_m_glob, y_m_glob, val_m_glob, val_m_glob, dims, dx, dy, lx, ly, rank, comm_cart)

# gather global grid values
grid1_glob = if rank == 0
    zeros(Tuple(dims[1:2]) .* size(val_grid1))
else
    zeros(0, 0)
end
grid2_glob = if rank == 0
    zeros(Tuple(dims[1:2]) .* size(val_grid2))
else
    zeros(0, 0)
end
grid3_glob = if rank == 0
    zeros(Tuple(dims[1:2]) .* size(val_grid3))
else
    zeros(0, 0)
end
gather!(Array(val_grid1), grid1_glob; root=0)
gather!(Array(val_grid2), grid2_glob; root=0)
gather!(Array(val_grid3), grid3_glob; root=0)

if rank == 0
    grid1_glob = create_grid_view(grid1_glob, dims, Nx, Ny, size(val_grid1, 1), size(val_grid1, 2))
    grid2_glob = create_grid_view(grid2_glob, dims, Nx, Ny, size(val_grid2, 1), size(val_grid2, 2))
    grid3_glob = create_grid_view(grid3_glob, dims, Nx, Ny, size(val_grid3, 1), size(val_grid3, 2))
end

finalize_global_grid(; finalize_MPI=false)


# 2) -------- COMPUTE THE SAME WITH 1 PROCESS --------
Nx = 18
Ny = 14
rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx + 10, Ny + 10, 1; init_MPI=false, quiet=true) # just init with Nx,Ny big enough to ensure no overlap
grid = ImplicitGlobalGrid.get_global_grid()

if rank == 0
    val_grid1_sp = @zeros(Nx - 1, Ny - 1)
    val_grid2_sp = @zeros(Nx, Ny)
    val_grid3_sp = @zeros(Nx + 1, Ny)
    # helper arrays, just as large as needed
    val_wt_sum = @ones(Nx + 1, Ny)
    wt_sum = @ones(Nx + 1, Ny)
    # interpolate to grid
    x_m_glob = Data.Array(x_m_glob)
    y_m_glob = Data.Array(y_m_glob)
    val_m_glob = Data.Array(val_m_glob)
    bilinearMarkerToGrid!(x_grid1_min, y_grid1_min, val_grid1_sp, x_m_glob, y_m_glob, val_m_glob, dx, dy, val_wt_sum, wt_sum, grid)
    bilinearMarkerToGrid!(x_grid2_min, y_grid2_min, val_grid2_sp, x_m_glob, y_m_glob, val_m_glob, dx, dy, val_wt_sum, wt_sum, grid)
    bilinearMarkerToGrid!(x_grid3_min, y_grid3_min, val_grid3_sp, x_m_glob, y_m_glob, val_m_glob, dx, dy, val_wt_sum, wt_sum, grid)

    # test approximate equality
    @test all(Array(val_grid1_sp) .≈ grid1_glob)
    @test all(Array(val_grid2_sp) .≈ grid2_glob)
    @test all(Array(val_grid3_sp) .≈ grid3_glob)

end


finalize_global_grid()
