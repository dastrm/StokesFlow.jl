using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = ENV["USE_GPU"] == "true" ? true : false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

include("../scripts/StokesSolver_multixpu.jl")
include("../scripts/GlobalGather.jl")

using Test, ImplicitGlobalGrid
import MPI


MPI.Init()
comm_size = MPI.Comm_size(MPI.COMM_WORLD)

# set up the global grid: it should always be identical
if comm_size == 1
    dimx = [1]
    dimy = [1]
    nx_arr = [26]
    ny_arr = [32]
elseif comm_size == 2
    dimx = [1, 2]
    dimy = [2, 1]
    nx_arr = [26, 14]
    ny_arr = [17, 32]
elseif comm_size == 3
    dimx = [1, 3]
    dimy = [3, 1]
    nx_arr = [26, 10]
    ny_arr = [12, 32]
elseif comm_size == 4
    dimx = [4, 2]
    dimy = [1, 2]
    nx_arr = [8, 14]
    ny_arr = [32, 17]
else
    @assert false "must be called with 1, 2, 3 or 4 MPI processes"
end

@assert all(dimx .* dimy .== comm_size)

# computed reference values: on 1 CPU process and very small ϵ
test_inds = [10, 75, 86, 119, 121, 134, 214, 242, 299, 388, 390, 456, 488, 530, 534, 547, 573, 586, 704, 708]
vx_ref = [41.73173769683686, 6.142863111007382, 0.0, 29.70649161399109, 32.70394707052636, -0.40328716034240464, 8.687574637478079, 7.86005318367501, 4.93933776612636, 1.0110898433129551, 0.375427760482275, -2.3702021866104905, -4.314218228770063, -0.015925503638153597, 0.0, -6.340779490641143, -9.51028612747028, -0.11201872079935975, -9.704484372789631, -20.664496394801567]
vy_ref = [-1.4207982072566916, 2.0287293488635845, -9.44595681821131, 6.0323304855365505, 9.225321330261993, -5.987233063528919, -5.166527470232467, -7.6118558217961185, -53.10019017148971, 6.905581051403612, 16.64205996892714, 1.0117881183567317, -55.41033606494321, 22.607255606607584, 13.168932892793151, -13.987289696185622, -21.516329272495312, 18.0221758504846, -35.087928602337946, -14.013224031900812]

# loop over each configuration of the global grid and test it
for i = eachindex(dimx)

    Nx = nx_arr[i]
    Ny = ny_arr[i]
    rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx, Ny, 1; dimx=dimx[i], dimy=dimy[i], dimz=1, init_MPI=false, quiet=true)

    @test nx_g() == 26 && ny_g() == 32

    dx = 0.5
    dy = 0.4
    lxglob = dx * (nx_g() - 1)
    lyglob = dy * (ny_g() - 1)
    xmid = 0.5lxglob
    ymid = 0.5lyglob

    # define global grid coordinates
    x0 = coords[1] * (Nx - 2) * dx
    y0 = coords[2] * (Ny - 2) * dy
    x = [(ix - 1) * dx for ix = 1:Nx] .+ x0
    y = [(iy - 1) * dy for iy = 1:Ny] .+ y0
    x_p = [(ix - 1) * dx + 0.5dx for ix = 1:Nx-1] .+ x0
    y_p = [(iy - 1) * dy + 0.5dy for iy = 1:Ny-1] .+ y0
    x_ρ = [(ix - 1) * dx - 0.5dx for ix = 1:Nx+1] .+ x0
    y_ρ = [(iy - 1) * dy for iy = 1:Ny] .+ y0

    # set initial conditions for ρ_vy, μ_b, μ_p
    ρ_vy = Data.Array([3.3 * exp(-0.01 * ((x_ρ[ix] - 1.1xmid)^2 + (y_ρ[iy] - 1.3ymid)^2)) for ix = 1:Nx+1, iy = 1:Ny])
    μ_b = Data.Array([exp(-0.025 * ((x[ix] - 1.1xmid)^2 + (y[iy] - 1.3ymid)^2)) for ix = 1:Nx, iy = 1:Ny])
    μ_p = Data.Array([exp(-0.025 * ((x_p[ix] - 1.1xmid)^2 + (y_p[iy] - 1.3ymid)^2)) for ix = 1:Nx-1, iy = 1:Ny-1])

    # define all other needed arrays
    P = @zeros(Nx - 1, Ny - 1)
    Vx = @zeros(Nx + 2, Ny + 1)
    Vy = @zeros(Nx + 1, Ny + 2)
    τxx = @zeros(Nx - 1, Ny - 1)
    τyy = @zeros(Nx - 1, Ny - 1)
    τxy = @zeros(Nx, Ny)
    ∇V = @zeros(Nx - 1, Ny - 1)
    dτPt = @zeros(Nx - 1, Ny - 1)
    Rx = @zeros(Nx - 2, Ny - 1)
    Ry = @zeros(Nx - 1, Ny - 2)
    dVxdτ = @zeros(Nx - 2, Ny - 1)
    dVydτ = @zeros(Nx - 1, Ny - 2)
    dτVx = @zeros(Nx - 2, Ny - 1)
    dτVy = @zeros(Nx - 1, Ny - 2)
    Vx_small = @zeros(Nx, Ny + 1)
    Vy_small = @zeros(Nx + 1, Ny)

    # more values
    dt = 0.0
    maxdisp = 0.5
    g_y = 9.81

    # call the solver
    dt, T_eff, runtime = solveStokes!(P, Vx, Vy, ρ_vy, μ_b, μ_p,
        τxx, τyy, τxy, ∇V, dτPt, Rx, Ry, dVxdτ, dVydτ, dτVx, dτVy, Vx_small, Vy_small,
        g_y, dx, dy, Nx, Ny,
        dt, maxdisp, comm_cart; use_free_surface_stabilization=true,
        ϵ=1e-8,
        print_info=false)


    @test dt ≈ maxdisp * MPI.Allreduce(min(dx / maximum(abs.(Vx[2:end-1, :])), dy / maximum(abs.(Vy[:, 2:end-1]))), MPI.MIN, comm_cart)

    # gather Vx, Vy in arrays that look as if a single process computed them
    Vx_glob, Vy_glob = gather_V_grid(Array(Vx), Array(Vy), rank, dims, Nx, Ny)

    if rank == 0
        # Reference Test
        # TODO: these two tests fail, but only with BOTH --check-bounds=true and USE_GPU==true
        #       this is very strange...
        @test Vx_glob[test_inds] ≈ vx_ref
        @test Vy_glob[test_inds] ≈ vy_ref
        # Test BC of Vx
        @test all(Vx_glob[1, :] .== -Vx_glob[3, :])
        @test all(Vx_glob[2, :] .== 0.0)
        @test all(Vx_glob[end, :] .== -Vx_glob[end-2, :])
        @test all(Vx_glob[end-1, :] .== 0.0)
        @test all(Vx_glob[:, 1] .== Vx_glob[:, 2])
        @test all(Vx_glob[:, end] .== Vx_glob[:, end-1])
        # Test BC of Vy
        @test all(Vy_glob[:, 1] .== -Vy_glob[:, 3])
        @test all(Vy_glob[:, 2] .== 0.0)
        @test all(Vy_glob[:, end] .== -Vy_glob[:, end-2])
        @test all(Vy_glob[:, end-1] .== 0.0)
        @test all(Vy_glob[1, :] .== Vy_glob[2, :])
        @test all(Vy_glob[end, :] .== Vy_glob[end-1, :])
    end

    finalize_global_grid(; finalize_MPI=false)

end

MPI.Finalize()
