USE_GPU = ENV["USE_GPU"] == "true" ? true : false

include("../scripts/StokesFlow2D_multixpu.jl")
include("../scripts/GlobalGather.jl")

using Test, ImplicitGlobalGrid
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
    dimx = [2]
    dimy = [2]
    nx_arr = [11]
    ny_arr = [14]
elseif comm_size == 6
    dimx = [2, 3]
    dimy = [3, 2]
    nx_arr = [11, 8]
    ny_arr = [10, 14]
elseif comm_size == 9
    dimx = [3]
    dimy = [3]
    nx_arr = [8]
    ny_arr = [10]
else
    @assert false "must be called with 1, 2, 3 or 4, 6 or 9 MPI processes"
end

@assert all(dimx .* dimy .== comm_size)

const Nx_glob = 20
const Ny_glob = 26

# prepare common arguments for Stokesflow2D called
const Nt = 5
const Lx_glob = 10
const Ly_glob = 10

const μ_air, μ_matrix, μ_plume = 1e-1, 1e0, 5e-1  # Viscosity
const ρ_air, ρ_matrix, ρ_plume = 1e-2, 3.3, 3.2   # Density
const plume_x, plume_y = Lx_glob / 2, Ly_glob / 2 # plume midpoint
const plume_r = min(Lx_glob, Ly_glob) / 5         # plume radius
const air_height = 0.2 * Ly_glob                  # height of the 'sticky air' layer on top
function density(x, y)
    if y < air_height
        ρ_air
    elseif (x - plume_x)^2 + (y - plume_y)^2 < plume_r^2
        ρ_plume
    else
        ρ_matrix
    end
end
function viscosity(x, y)
    if y < air_height
        μ_air
    elseif (x - plume_x)^2 + (y - plume_y)^2 < plume_r^2
        μ_plume
    else
        μ_matrix
    end
end

# define reference values (computed by one process with very low ϵ)
inds = [12, 64, 136, 151, 155, 169, 214, 467, 502, 531]
vxref = [-0.030598585657308882, -0.03971445331323613, -0.19832594566486283, 0.19832594566486786, 0.10567079100073508, 0.42658320978610326, 0.38623283329846403, 0.14878040649997637, -0.14268132666324265, 0.05032230621463761]
vyref = [-0.04351872565396834, -0.0690332491743087, -0.005605609488209911, 0.06568832004492553, -0.04425718809923243, 0.15316129230490624, 0.271405611687118, 0.07634014951042657, 0.10492572325942193, 0.0055823677070680104]

# loop over each configuration of the global grid and test it
for i = eachindex(dimx)

    # call the StokesFlow function
    Nx = nx_arr[i]
    Ny = ny_arr[i]
    Vx_glob, Vy_glob, t_tot = StokesFlow2D(Nt, Nx, Ny, Lx_glob, Ly_glob, density, viscosity;
        dimx=dimx[i], dimy=dimy[i], stokes_ϵ=1e-8,
        RAND_MARKER_POS=false, plot_fields_live=false, plot_markers_live=false, save_to_file=false, print_info=false, init_MPI=false, finalize_MPI=false)

    if size(Vx_glob, 1) > 0 # implies rank == 0

        # test array sizes
        @test all(size(Vx_glob) .== (Nx_glob + 2, Ny_glob + 1))
        @test all(size(Vy_glob) .== (Nx_glob + 1, Ny_glob + 2))

        # compare with reference values
        @test all(isapprox(Vx_glob[inds], vxref; atol=1e-6))
        @test all(isapprox(Vy_glob[inds], vyref; atol=1e-6))

    end

end

MPI.Finalize()
