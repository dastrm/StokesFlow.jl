USE_GPU = ENV["USE_GPU"] == "true" ? true : false

include("../scripts/common.jl")
include("../scripts/GlobalGather.jl")

using Random, Test, ImplicitGlobalGrid
import MPI

ENV["GKSwstype"] = "nul"

if !MPI.Initialized()
    MPI.Init()
end
comm_size = MPI.Comm_size(MPI.COMM_WORLD)

# set up the global grid: it should always be identical
if comm_size == 1
    dimx = [1]
    dimy = [1]
    nx_arr = [20]
    ny_arr = [26]
elseif comm_size == 2
    dimx = [1, 2]
    dimy = [2, 1]
    nx_arr = [20, 11]
    ny_arr = [14, 26]
elseif comm_size == 3
    dimx = [1, 3]
    dimy = [3, 1]
    nx_arr = [20, 8]
    ny_arr = [10, 26]
elseif comm_size == 4
    dimx = [1]
    dimy = [4]
    nx_arr = [20]
    ny_arr = [8]
elseif comm_size == 6
    dimx = [2, 3]
    dimy = [3, 2]
    nx_arr = [11, 8]
    ny_arr = [10, 14]
elseif comm_size == 8
    dimx = [2]
    dimy = [4]
    nx_arr = [11]
    ny_arr = [8]
elseif comm_size == 9
    dimx = [3]
    dimy = [3]
    nx_arr = [8]
    ny_arr = [10]
else
    @assert false "must be called with 1, 2, 3 or 4, 6, 8 or 9 MPI processes"
end

@assert all(dimx .* dimy .== comm_size)

const Nx_glob = 20
const Ny_glob = 26

# prepare common arguments for Stokesflow2D called
const Nt = 5
const Lx_glob = 10
const Ly_glob = 10

# loop over each configuration of the global grid and test it
for i = eachindex(dimx)

    Nx = nx_arr[i]
    Ny = ny_arr[i]

    rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1; dimx=dimx[i], dimy=dimy[i], dimz=1, quiet=true, init_MPI=false)

    # derived quantities
    dx = Lx_glob / (nx_g() - 1)
    dy = Ly_glob / (ny_g() - 1)
    lx = (Nx - 1) * dx
    ly = (Ny - 1) * dy
    x0 = coords[1] * (Nx - 2) * dx
    y0 = coords[2] * (Ny - 2) * dy

    marker_density = 3

    x_m, y_m, ρ_m, μ_m = initializeMarkersCPU(comm_cart, dims, coords, marker_density, lx, ly, dx, dy, Nx, Ny, false)
    @test allequal((size(x_m, 1), size(y_m, 1), size(ρ_m, 1), size(μ_m, 1)))

    x_m_glob = zeros(0)
    y_m_glob = zeros(0)
    ρ_m_glob = zeros(0)
    μ_m_glob = zeros(0)
    gather_markers!(x_m, y_m, ρ_m, μ_m, x_m_glob, y_m_glob, ρ_m_glob, μ_m_glob, dims, dx, dy, lx, ly, rank, comm_cart)


    if rank == 0

        # sort the gathered markers in preparation for comparison
        perm = sortperm(x_m_glob .* Lx_glob .* 2 .+ y_m_glob)
        x_m_glob = x_m_glob[perm]
        y_m_glob = y_m_glob[perm]

        # manually create marker positions as the should be
        Nmx = marker_density * (Nx_glob - 1)
        Nmy = marker_density * (Ny_glob - 1)
        dx = Lx_glob / (Nx_glob - 1)
        dy = Ly_glob / (Ny_glob - 1)
        dxm = dx / marker_density
        dym = dy / marker_density
        xcoords = LinRange(0.5dxm, Lx_glob - 0.5dxm, Nmx)
        ycoords = LinRange(0.5dym, Ly_glob - 0.5dym, Nmy)
        Nm = Nmx * Nmy
        x_m_ref = zeros(Nm)
        y_m_ref = zeros(Nm)
        m = 1
        for ix = 1:Nmx
            for iy = 1:Nmy
                x_m_ref[m] = xcoords[ix]
                y_m_ref[m] = ycoords[iy]
                m += 1
            end
        end
        # sort the reference markers as well
        perm = sortperm(x_m_ref .* Lx_glob .* 2 .+ y_m_ref)
        x_m_ref = x_m_ref[perm]
        y_m_ref = y_m_ref[perm]

        @test Nm == size(x_m_glob, 1)
        @test Nm == size(y_m_glob, 1)

        @test all(x_m_glob ≈ x_m_ref)
        @test all(y_m_glob ≈ y_m_ref)

    end

    finalize_global_grid(; finalize_MPI=false)

end
