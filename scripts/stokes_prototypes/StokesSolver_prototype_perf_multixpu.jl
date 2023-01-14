const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, Test, ImplicitGlobalGrid, MPI

include("../GlobalGather.jl")

# Unfortunately, the extended Vx and Vy arrays are difficult to handle with ImplicitGlobalGrid.
# Since update_halo!(..) can only update the first and last entries in each dimension, even with a bigger overlap,
# there have to be kept two separate arrays: First, the unpadded Vx / Vy arrays must be used for the iterations and
# application of zero / no-flux BC.
# (On padded arrays, the zero BC would be applied to indices 2 and end-1, which are not updated in update_halo!(...))
# Then, the entire Vx / Vy arrays are copied into the padded arrays Vx_pad / Vy_pad, to which the artificial
# 'zero second derivative' BC are applied, and a final halo update is performed to have these BC only valid at
# the physical boundaries.
# This is possible since these artificial BC are not relevant in the iterations, only for markers.

@views function Stokes2D()
    # Physics
    lx, ly = [100000, 100000] ./ 100000 # domain size
    μ_air, μ_matrix, μ_plume = [1e17, 1e19, 1e18] ./ 1e19 # Viscosity
    ρ_air, ρ_matrix, ρ_plume = [1, 3300, 3200] ./ 1000 # Density
    g_y = 9.81
    plume_x, plume_y = lx / 2, ly / 2 # plume midpoint
    plume_r = 20000 / 100000 # plume radius
    air_height = 0.2 * ly

    use_free_surface_stabilization = true
    maxdisp = 0.5

    # Numerics
    # !!! THIS IS CURRENTLY SET UP FOR 6 PROCESSES !!!
    nx, ny = 44, 65#128,128
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, 1; init_MPI=true, dimx=3)

    # Derived Numerics
    dx, dy = lx / (nx_g() - 1), ly / (ny_g() - 1)

    # Array allocations
    P = @zeros(nx - 1, ny - 1)
    τxx = @zeros(nx - 1, ny - 1) #p-nodes
    τyy = @zeros(nx - 1, ny - 1) #p-nodes
    Vx = @zeros(nx, ny + 1)
    Vy = @zeros(nx + 1, ny)
    Vx_pad = @zeros(nx + 2, ny + 1)
    Vy_pad = @zeros(nx + 1, ny + 2)
    τxy = @zeros(nx, ny)

    # *global* coordinates for points
    shiftx = coords[1] * (nx - 2) * dx
    shifty = coords[2] * (ny - 2) * dy
    x = [(ix - 1) * dx + shiftx for ix = 1:nx] # basic nodes = τ_xy-nodes
    y = [(iy - 1) * dy + shifty for iy = 1:ny]
    x_p = [(ix - 1) * dx + 0.5dx + shiftx for ix = 1:nx-1] # pressure nodes
    y_p = [(iy - 1) * dy + 0.5dy + shifty for iy = 1:ny-1]
    x_vx = [(ix - 2) * dx + shiftx for ix = 1:nx+2] # Vx_pad nodes
    y_vx = [(iy - 1) * dy - 0.5dy + shifty for iy = 1:ny+1]
    x_vy = [(ix - 1) * dx - 0.5dx + shiftx for ix = 1:nx+1] # Vy nodes
    y_vy = [(iy - 2) * dy + shifty for iy = 1:ny+2]
    x_τxy = x
    y_τxy = y
    x_ρ = x_vy
    y_ρ = y_vy[2:end-1]

    # set density & viscosity
    ρ_vy = zeros(nx + 1, ny) # on Vy-nodes
    μ_b = zeros(nx, ny) # on τxy-nodes
    μ_p = zeros(nx - 1, ny - 1) # on P-nodes (also τxx-, τyy-nodes)
    set_properties!(x_p, y_p, μ_p, plume_x, plume_y, plume_r, μ_matrix, μ_plume, μ_air, air_height) # viscosity: P-nodes
    set_properties!(x_τxy, y_τxy, μ_b, plume_x, plume_y, plume_r, μ_matrix, μ_plume, μ_air, air_height) # viscosity: τxy-nodes
    set_properties!(x_ρ, y_ρ, ρ_vy, plume_x, plume_y, plume_r, ρ_matrix, ρ_plume, ρ_air, air_height) # density: Vy-nodes
    ρ_vy = Data.Array(ρ_vy)
    μ_b = Data.Array(μ_b)
    μ_p = Data.Array(μ_p)

    #=
    # plot density & viscosity
    p1 = heatmap(x_p  ,y_p  ,Array(μ_p)'  ,yflip=true,title="Viscosity μ_p"  ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p2 = heatmap(x_τxy,y_τxy,Array(μ_b)',yflip=true,title="Viscosity μ_b",xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p3 = heatmap(x_vy ,y_vy ,Array(ρ_vy)' ,yflip=true,title="Density ρ_vy"   ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    display(plot(p1,p2,p3))
    #display(plot(p1)); display(plot(p2)); display(plot(p3))
    =#

    # more arrays
    ∇V = @zeros(nx - 1, ny - 1)
    dτP = @zeros(nx - 1, ny - 1)
    Rx = @zeros(nx - 2, ny - 1)
    Ry = @zeros(nx - 1, ny - 2)
    dVxdτ = @zeros(nx - 2, ny - 1)
    dVydτ = @zeros(nx - 1, ny - 2)
    dτVx = @zeros(nx - 2, ny - 1)
    dτVy = @zeros(nx - 1, ny - 2)
    Vdmp = 4.0
    Vsc = 0.45#1.0         # relaxation parameter for the momentum equations pseudo-timesteps limiters
    Ptsc = 1.0 / 4.0 * 2.0
    min_dxy2 = min(dx, dy)^2
    max_nxy = max(nx, ny)
    dampX = 1.0 - Vdmp / nx # damping term for the x-momentum equation
    dampY = 1.0 - Vdmp / ny # damping term for the y-momentum equation

    # numerical helper values
    _dx = 1.0 / dx
    _dy = 1.0 / dy

    # PT setup and iterations
    @parallel compute_timesteps!(dτVx, dτVy, dτP, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    ncheck = min(5 * min(nx_g(), ny_g()), 2000)
    ndtupdate = 10 * min(nx_g(), ny_g())
    dt = 0.0
    t1 = 0
    itert1 = 11
    ϵ = 1e-9 # tol
    err_evo1 = []
    err_evo2 = []
    err = 2ϵ
    iter = 1
    niter = 0
    iterMax = 100000
    while err > ϵ && iter <= iterMax
        if (iter == itert1)
            t1 = Base.time()
        end

        @parallel compute_P_τ!(∇V, P, Vx, Vy, dτP, τxx, τyy, τxy, μ_p, μ_b, _dx, _dy)
        @parallel compute_dV!(dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, _dx, _dy, dt)
        @hide_communication (8, 8) begin
            @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
            # Free slip BC
            @parallel (1:size(Vx, 1)) bc_y_noflux!(Vx)
            @parallel (1:size(Vx, 2)) bc_x_zero!(Vx)
            @parallel (1:size(Vy, 1)) bc_y_zero!(Vy)
            @parallel (1:size(Vy, 2)) bc_x_noflux!(Vy)
            update_halo!(Vx, Vy)
        end

        if use_free_surface_stabilization && (iter % ndtupdate == 0)
            dt = compute_dt(Vx, Vy, maxdisp, dx, dy, comm_cart)
        end

        if iter % ncheck == 0
            if me == 0
                @show iter
            end
            dt_check = if use_free_surface_stabilization
                compute_dt(Vx, Vy, maxdisp, dx, dy, comm_cart)
            else
                0.0
            end
            @parallel compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, _dx, _dy, dt_check)
            err = compute_err(Rx, Ry, ∇V, comm_cart)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            #@printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_∇V=%1.3e] \n", iter, err, mean_Rx, mean_Ry, mean_∇V)
        end

        iter += 1
        niter += 1
    end
    t2 = Base.time()
    t_it = (t2 - t1) / (niter - itert1 + 1)
    A_eff = (3 * 2) / 1e9 * nx * ny * sizeof(Data.Number)
    T_eff = A_eff / t_it
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, t2 - t1, round(T_eff, sigdigits=2))

    #=
    # Visualization
    p1 = heatmap(x_p ,  y_p, Array(P)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Pressure")
    p2 = heatmap(x_vy, y_vy, Array(Vy)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vy")
    #p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:inferno, title="log10(Ry)")
    p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    p6 = heatmap(x_vx, y_vx, Array(Vx)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vx")
    #display(plot(p1, p2, p4, p5))
    display(plot(p1,p2,p5,p6))
    =#


    # Since update_halo!() from ImplicitGlobalGrid can only update the first/last index in any dimension,
    # there must be kept a separate array to apply the final "zero second derivative" BC on the physical domain boundaries.
    # This artificial BC is necessary for the velocity interpolation to markers.
    Vx_pad[2:end-1, :] .= Vx
    Vy_pad[:, 2:end-1] .= Vy
    @parallel (1:size(Vx_pad, 2)) bc_x_mirror!(Vx_pad)
    @parallel (1:size(Vy_pad, 1)) bc_y_mirror!(Vy_pad)
    update_halo!(Vx_pad, Vy_pad)


    # Assemble the return values as if a single process computed them, for tests
    Vx_glob, Vy_glob = gather_V_grid(Vx_pad, Vy_pad, me, dims, nx, ny)

    return Vx_glob, Vy_glob, me
end



function compute_err(Rx, Ry, ∇V, comm)
    comm_size = MPI.Comm_size(comm)
    mean_Rx = MPI.Allreduce(mean(abs.(Rx)), MPI.SUM, comm) / comm_size
    mean_Ry = MPI.Allreduce(mean(abs.(Ry)), MPI.SUM, comm) / comm_size
    mean_∇V = MPI.Allreduce(mean(abs.(∇V)), MPI.SUM, comm) / comm_size
    return max(mean_Rx, mean_Ry, mean_∇V)
end

# this function is very slow and could be improved significantly, but is not often called
function compute_dt(Vx, Vy, maxdisp, dx, dy, comm)
    min_l = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)))
    return maxdisp * MPI.Allreduce(min_l, MPI.MIN, comm)
end

@parallel function compute_timesteps!(dτVx, dτVy, dτP, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    @all(dτVx) = Vsc * min_dxy2 / @av_xa(μ_p) / 4.1
    @all(dτVy) = Vsc * min_dxy2 / @av_ya(μ_p) / 4.1
    @all(dτP) = Ptsc * 4.1 * @all(μ_p) / max_nxy
    return
end

@parallel_indices (ix, iy) function compute_P_τ!(∇V, P, Vx, Vy, dτP, τxx, τyy, τxy, μ_p, μ_b, _dx, _dy)
    xmax, ymax = size(τxy)
    if ix <= xmax && iy <= ymax
        # read V arrays
        vx0 = Vx[ix, iy]
        vx1 = Vx[ix, iy+1]
        vy0 = Vy[ix, iy]
        vy1 = Vy[ix+1, iy]
        # update τxx, τxx, P, ∇V
        if ix < xmax && iy < ymax # size of P-nodes is 1 smaller than basic nodes (τxy) in each dimension
            # read additional V
            vx2 = Vx[ix+1, iy+1]
            vy2 = Vy[ix+1, iy+1]
            # update
            dVx_dx = (vx2 - vx1) * _dx
            dVy_dy = (vy2 - vy1) * _dy
            div_V = dVx_dx + dVy_dy
            ∇V[ix, iy] = div_V
            P[ix, iy] -= dτP[ix, iy] * div_V
            μ = μ_p[ix, iy]
            τxx[ix, iy] = 2.0 * μ * (dVx_dx - 1.0 / 3.0 * div_V)
            τyy[ix, iy] = 2.0 * μ * (dVy_dy - 1.0 / 3.0 * div_V)
        end
        # update τxy
        dVx_dy = (vx1 - vx0) * _dy
        dVy_dx = (vy1 - vy0) * _dx
        τxy[ix, iy] = 2.0 * μ_b[ix, iy] * 0.5 * (dVx_dy + dVy_dx)
    end
    return
end

@inline function compute_ResX(ix, iy, τxx, τxy, P, _dx, _dy)
    return ((τxx[ix+1, iy] - τxx[ix, iy]) * _dx
            +
            (τxy[ix+1, iy+1] - τxy[ix+1, iy]) * _dy
            -
            (P[ix+1, iy] - P[ix, iy]) * _dx)
end

@inline function compute_ResY(ix, iy, τyy, τxy, P, ρ_vy, Vx, Vy, g_y, _dx, _dy, dt)
    av_inn_y_Vx = 0.25 * (Vx[ix, iy+1] + Vx[ix+1, iy+1] + Vx[ix, iy+2] + Vx[ix+1, iy+2])
    d_xi_2_ρ_vy = ρ_vy[ix+2, iy+1] - ρ_vy[ix, iy+1]
    d_yi_2_ρ_vy = ρ_vy[ix+1, iy+2] - ρ_vy[ix+1, iy]
    return ((τyy[ix, iy+1] - τyy[ix, iy]) * _dy
            +
            (τxy[ix+1, iy+1] - τxy[ix, iy+1]) * _dx
            -
            (P[ix, iy+1] - P[ix, iy]) * _dy
            +
            g_y * (ρ_vy[ix+1, iy+1] - dt * (av_inn_y_Vx * d_xi_2_ρ_vy * 0.5 * _dx
                                            +
                                            Vy[ix+1, iy+1] * d_yi_2_ρ_vy * 0.5 * _dy)))
end

@parallel_indices (ix, iy) function compute_dV!(dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, _dx, _dy, dt)
    if ix <= size(dVxdτ, 1) && iy <= size(dVxdτ, 2)
        Rx_temp = compute_ResX(ix, iy, τxx, τxy, P, _dx, _dy)
        dVxdτ[ix, iy] = dampX * dVxdτ[ix, iy] + Rx_temp
    end
    if ix <= size(dVydτ, 1) && iy <= size(dVydτ, 2)
        Ry_temp = compute_ResY(ix, iy, τyy, τxy, P, ρ_vy, Vx, Vy, g_y, _dx, _dy, dt)
        dVydτ[ix, iy] = dampY * dVydτ[ix, iy] + Ry_temp
    end
    return
end

@parallel_indices (ix, iy) function compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
    if ix <= size(dτVx, 1) && iy <= size(dτVx, 2)
        Vx[ix+1, iy+1] += dτVx[ix, iy] * dVxdτ[ix, iy]
    end
    if ix <= size(dτVy, 1) && iy <= size(dτVy, 2)
        Vy[ix+1, iy+1] += dτVy[ix, iy] * dVydτ[ix, iy]
    end
    return
end

@parallel_indices (ix, iy) function compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, _dx, _dy, dt)
    if ix <= size(Rx, 1) && iy <= size(Rx, 2)
        Rx[ix, iy] = compute_ResX(ix, iy, τxx, τxy, P, _dx, _dy)
    end
    if ix <= size(Ry, 1) && iy <= size(Ry, 2)
        Ry[ix, iy] = compute_ResY(ix, iy, τyy, τxy, P, ρ_vy, Vx, Vy, g_y, _dx, _dy, dt)
    end
    return
end

# side boundaries
@parallel_indices (iy) function bc_x_zero!(A::Data.Array)
    A[1, iy] = 0.0
    A[end, iy] = 0.0
    return
end
@parallel_indices (iy) function bc_x_noflux!(A::Data.Array)
    A[1, iy] = A[2, iy]
    A[end, iy] = A[end-1, iy]
    return
end
@parallel_indices (iy) function bc_x_mirror!(A::Data.Array)
    A[1, iy] = -A[3, iy]
    A[end, iy] = -A[end-2, iy]
    return
end

# horizontal boundaries
@parallel_indices (ix) function bc_y_zero!(A::Data.Array)
    A[ix, 1] = 0.0
    A[ix, end] = 0.0
    return
end
@parallel_indices (ix) function bc_y_noflux!(A::Data.Array)
    A[ix, 1] = A[ix, 2]
    A[ix, end] = A[ix, end-1]
    return
end
@parallel_indices (ix) function bc_y_mirror!(A::Data.Array)
    A[ix, 1] = -A[ix, 3]
    A[ix, end] = -A[ix, end-2]
    return
end

# This function sets material properties in arr depending on where the node is
function set_properties!(x, y, arr, plume_x, plume_y, plume_r, matrix, plume, air, air_height)
    nx, ny = size(arr)
    @assert nx <= size(x, 1) && ny <= size(y, 1)
    for ix = 1:nx
        for iy = 1:ny
            val = 0.0
            if y[iy] < air_height
                val = air
            elseif (x[ix] - plume_x)^2 + (y[iy] - plume_y)^2 <= plume_r^2
                val = plume
            else
                val = matrix
            end
            arr[ix, iy] = val
        end
    end
    return nothing
end

#Stokes2D()

# run with 6 ranks
@testset "StokesPrototype_perf_multixpu" begin
    Vx_pad, Vy_pad, me = Stokes2D()
    #Vx_pad=zeros(130,129)
    #Vy_pad=zeros(129,130)
    #file1 = open("Vx.bin", "r"); read!(file1,Vx_pad); close(file1)
    #file2 = open("Vy.bin", "r"); read!(file2,Vy_pad); close(file2)
    if me == 0
        #file1 = open("Vx.bin", "w"); write(file1,Vx_pad); close(file1)
        #file2 = open("Vy.bin", "w"); write(file2,Vy_pad); close(file2)
        Vx = Vx_pad[2:end-1, :]
        Vy = Vy_pad[:, 2:end-1]
        indsx = [3, 56, 60, 90, 99]
        indsy = [28, 68, 95, 96, 127]
        #for 128
        refsX = [-0.00036803049758490355 0.00010277500628777059 0.00026451668768832286 0.0002654556171811493 0.0002427914398641037; -0.002833099685336914 0.0007306307934672929 0.0021771666617763975 0.0021096038062890965 0.0011727566013140771; -0.0015762231740737759 0.00039399374532432876 0.0012119200401637183 0.0011736588923177526 0.0006377976620771319; 0.005178056510938031 -0.0017164279966205213 -0.003785804956026086 -0.003737865431530065 -0.002596696719907533; 0.004658308398754078 -0.0013770926440414655 -0.0033519140416498274 -0.003335527066709783 -0.002604464726635115]
        refsY = [0.004517210079040068 0.007034594778397958 0.004210274869516819 0.004077602686545543 0.0001212234308027144; -0.005650928028720009 -0.010428003561303635 -0.005061741656495928 -0.0048563749107042534 -0.00012565956961902374; -0.006442760652637244 -0.011604449243728616 -0.005723916322913189 -0.005476337223473083 -0.00013752961326395758; -0.0007519710495368777 0.0008306568410141842 -0.0008040449541852101 -0.0008077371949626541 -3.416181915913092e-5; 0.0015815709663818817 0.003546874593750863 0.001331570819427396 0.001260273578007849 2.4848542745070437e-5]
        @test all(isapprox.(refsX, Vx[indsx, indsy]; atol=3e-6))
        @test all(isapprox.(refsY, Vy[indsx, indsy]; atol=3e-6))
        # BC of Vx
        @test all(Vx_pad[1, :] .== -Vx_pad[3, :])
        @test all(Vx_pad[2, :] .== 0.0)
        @test all(Vx_pad[end, :] .== -Vx_pad[end-2, :])
        @test all(Vx_pad[end-1, :] .== 0.0)
        @test all(Vx_pad[:, 1] .== Vx_pad[:, 2])
        @test all(Vx_pad[:, end] .== Vx_pad[:, end-1])
        # BC of Vy
        @test all(Vy_pad[:, 1] .== -Vy_pad[:, 3])
        @test all(Vy_pad[:, 2] .== 0.0)
        @test all(Vy_pad[:, end] .== -Vy_pad[:, end-2])
        @test all(Vy_pad[:, end-1] .== 0.0)
        @test all(Vy_pad[1, :] .== Vy_pad[2, :])
        @test all(Vy_pad[end, :] .== Vy_pad[end-1, :])
    end
end
