include("../scripts/multi_process/MarkerExchange.jl")
include("../scripts/GlobalGather.jl")

using Test, ImplicitGlobalGrid
import MPI


MPI.Init()
comm_size = MPI.Comm_size(MPI.COMM_WORLD)

# set up the global grid: it should always be identical
if comm_size == 1
    dimx=[1]; dimy=[1]; nx_arr=[26]; ny_arr=[32];
elseif comm_size == 2
    dimx=[1,2]; dimy=[2,1]; nx_arr=[26,14]; ny_arr=[17,32];
elseif comm_size == 3
    dimx=[1,3]; dimy=[3,1]; nx_arr=[26,10]; ny_arr=[12,32];
elseif comm_size == 4
    dimx=[4,2]; dimy = [1,2]; nx_arr=[8,14]; ny_arr=[32,17];
elseif comm_size == 6
    dimx=[2,3]; dimy = [3,2]; nx_arr=[14,10]; ny_arr=[12,17];
elseif comm_size == 8
    dimx=[4]; dimy = [2]; nx_arr=[8]; ny_arr=[17];
elseif comm_size == 9
    dimx=[3]; dimy = [3]; nx_arr=[10]; ny_arr=[12];
else
    @assert false "must be called with 1, 2, 3 or 4, 6, 8 or 9 MPI processes"
end

@assert all(dimx .* dimy .== comm_size)

function l(N,d)
    return 0.005d + 0.49d*rand()
end
function h(N,d)
    return (N-1)*d-(0.005d + 0.49d*rand())
end
function m(N,d)
    return 0.505d + ((N-3)*d + 0.99d)*rand()
end

function init_markers(Nx,Ny,dx,dy)
    # each process gets 17 markers, placed such that one must be moved to each of the 8 neighbouring processes (if they exist),
    # and all other markers should stay
    x_m = zeros(0); y_m = zeros(0)
    append!(x_m, [l(Nx,dx), l(Nx,dx), l(Nx,dx)])
    append!(y_m, [l(Ny,dy), m(Ny,dy), h(Ny,dy)])

    append!(x_m, [l(Nx,dx)      , l(Nx,dx), l(Nx,dx)      ].+0.5dx)
    append!(y_m, [l(Ny,dy)+0.5dy, m(Ny,dy), h(Ny,dy)-0.5dy]       )

    append!(x_m, [m(Nx,dx), m(Nx,dx)      , m(Nx,dx), m(Nx,dx)      , m(Nx,dx)])
    append!(y_m, [l(Ny,dy), l(Ny,dy)+0.5dy, m(Ny,dy), h(Ny,dy)-0.5dy, h(Ny,dy)])

    append!(x_m, [h(Nx,dx)      , h(Nx,dx), h(Nx,dx)      ].-0.5dx)
    append!(y_m, [l(Ny,dy)+0.5dy, m(Ny,dy), h(Ny,dy)-0.5dy]       )

    append!(x_m, [h(Nx,dx), h(Nx,dx), h(Nx,dx)])
    append!(y_m, [l(Ny,dy), m(Ny,dy), h(Ny,dy)])

    return x_m, y_m, rand(size(x_m,1)).*1000, rand(size(x_m,1)).*100000
end

function create_sort_criterium(x_m, y_m, lx_loc, comm_size)
    return x_m.*lx_loc.*comm_size.+y_m
end

# loop over each configuration of the global grid and test it
for i=eachindex(dimx)

    Nx = nx_arr[i]
    Ny = ny_arr[i]
    rank, dims, nprocs, coords, comm_cart = init_global_grid(Nx,Ny,1;dimx=dimx[i],dimy=dimy[i],dimz=1,init_MPI=false,quiet=true)

    @test nx_g() == 26 && ny_g() == 32

    dx = 0.5
    dy = 0.4
    lx_loc = (Nx-1)*dx
    ly_loc = (Ny-1)*dy

    # create empty arrays for markers
    x_m_glob1 = zeros(0); y_m_glob1 = zeros(0); ρ_m_glob1 = zeros(0); μ_m_glob1 = zeros(0)
    x_m_glob2 = zeros(0); y_m_glob2 = zeros(0); ρ_m_glob2 = zeros(0); μ_m_glob2 = zeros(0)
    x_m       = zeros(0); y_m       = zeros(0); ρ_m       = zeros(0); μ_m = zeros(0)

    # initialize markers (random, but inside certain regions). their positions must be unique, since they have to be sorted for later tests
    unique_positions = false
    while !unique_positions
        x_m, y_m, ρ_m, μ_m = init_markers(Nx,Ny,dx,dy)
        gather_markers!(x_m,y_m,ρ_m,μ_m,x_m_glob1,y_m_glob1,ρ_m_glob1,μ_m_glob1,dims,dx,dy,lx_loc,ly_loc,rank,comm_cart)
        if rank == 0
            # sort markers according to some criterium
            sort_crit = create_sort_criterium(x_m_glob1, y_m_glob1, lx_loc, comm_size)
            sorted_m = sort(sort_crit)        
            # check uniqueness of sorting. is necessary for tests, but might be (very unlikely) not the case
            # because of random marker initialization.
            unique_positions = (minimum(diff(sorted_m)) > 1e-12 * maximum(sort_crit))
        end
        unique_positions = Bool(MPI.Allreduce([Int(unique_positions)],MPI.LOR, comm_cart)[1])
    end

#=
    # uncomment to show local marker positions
    using Plots
    vline( [0.5dx,dx,(Nx-2)*dx,(Nx-1.5)*dx])
    hline!([0.5dy,dy,(Ny-2)*dy,(Ny-1.5)*dy])
    display(scatter!(x_m,y_m,legend=false,xlims=(0,(Nx-1)*dx),ylims=(0,(Ny-1)*dy)))
=#

    # exchange markers
    x_m, y_m, ρ_m, μ_m = exchangeMarkers!(comm_cart, dims, [lx_loc, ly_loc], dx, dy, x_m, y_m, ρ_m, μ_m)

    # gather markers after exchange
    gather_markers!(x_m,y_m,ρ_m,μ_m,x_m_glob2,y_m_glob2,ρ_m_glob2,μ_m_glob2,dims,dx,dy,lx_loc,ly_loc,rank,comm_cart)


    # TEST 1: all markers must be inside local bounds
    # low x
    if coords[1] == 0
        @test all(x_m .> 0)
    else
        #@test all(x_m .> 0.5dx)
    end
    # high x
    if coords[1] == dims[1]-1
        @test all(x_m .< (Nx-1)*dx)
    else
        #@test all(x_m .< (Nx-1)*dx -0.5dx)
    end
    # low y
    if coords[2] == 0
        @test all(y_m .> 0)
    else
        @test all(y_m .> 0.5dy)
    end
    # high y
    if coords[2] == dims[2]-1
        @test all(y_m .< (Ny-1)*dy)
    else
        @test all(y_m .< (Ny-1)*dy -0.5dy)
    end

    # TEST 2: all markers still exist and have correct coordinates
    if rank == 0

        # check sizes
        @test size(x_m_glob1,1) == 17*comm_size
        @test size(x_m_glob2,1) == 17*comm_size

        @test size(y_m_glob1,1) == 17*comm_size
        @test size(y_m_glob2,1) == 17*comm_size

        @test size(ρ_m_glob1,1) == 17*comm_size
        @test size(ρ_m_glob2,1) == 17*comm_size

        @test size(μ_m_glob1,1) == 17*comm_size
        @test size(μ_m_glob2,1) == 17*comm_size

        # compare equality of sorted arrays
        order1 = sortperm(create_sort_criterium(x_m_glob1, y_m_glob1, lx_loc, comm_size))
        order2 = sortperm(create_sort_criterium(x_m_glob2, y_m_glob2, lx_loc, comm_size))

        @test all(x_m_glob1[order1] .≈ x_m_glob2[order2])
        @test all(y_m_glob1[order1] .≈ y_m_glob2[order2])
        @test all(ρ_m_glob1[order1] .== ρ_m_glob2[order2])
        @test all(μ_m_glob1[order1] .== μ_m_glob2[order2])

    end

    finalize_global_grid(;finalize_MPI=false)

end

MPI.Finalize()

