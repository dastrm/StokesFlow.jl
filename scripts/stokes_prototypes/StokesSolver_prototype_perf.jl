#=
This script solves the 2D Stokes & Continuity equations for given arrays of μ, ρ.

∂τxx/∂x + ∂τxy/∂y - ∂P/∂x = 0
∂τyx/∂x + ∂τyy/∂y - ∂P/∂y = -ρ*g*y
∂vx/∂x + ∂vy/∂y = 0
τxx = 2μ*∂vx/∂x
τyy = 2μ*∂vy/∂y
τxy = τyx = μ*(∂vx/∂y + ∂vy/∂x)

Additionally, dt is calculated as dt = maxdisp*min(dx/maximum(Vx),dy/maximum(Vy)),
and for the rhs of the y-Stokes equation, the density field ρ is implicitly advected
to the next timestep using Vx, Vy. ("Free surface stabilization")

Otherwise, PT iterations are implemented similarly to the 2D minapp of ParallelStencil.jl
https://github.com/omlins/ParallelStencil.jl/blob/main/miniapps/Stokes2D.jl
=#

#=
What's still missing:
- Performance (Kernel fusing, tuning relaxation parameters, ...)
  see e.g. https://github.com/PTsolvers/PseudoTransientStokes.jl/blob/main/scripts/Stokes2D_ve_perf.jl
=#

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, Test

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
    nx, ny = 127, 127#35, 45

    # Derived Numerics
    dx, dy = lx / (nx - 1), ly / (ny - 1)

    # Array allocations
    P = @zeros(nx - 1, ny - 1)
    τxx = @zeros(nx - 1, ny - 1) #p-nodes
    τyy = @zeros(nx - 1, ny - 1) #p-nodes
    Vx = @zeros(nx, ny + 1)
    Vy = @zeros(nx + 1, ny)
    τxy = @zeros(nx, ny)

    # coordinates for points
    x = [(ix - 1) * dx for ix = 1:nx] # basic nodes = τ_xy-nodes
    y = [(iy - 1) * dy for iy = 1:ny]
    x_p = [(ix - 1) * dx + 0.5dx for ix = 1:nx-1] # pressure nodes
    y_p = [(iy - 1) * dy + 0.5dy for iy = 1:ny-1]
    x_vx = x                               # Vx nodes
    y_vx = [(iy - 1) * dy - 0.5dy for iy = 1:ny+1]
    x_vy = [(ix - 1) * dx - 0.5dx for ix = 1:nx+1] # Vy nodes
    y_vy = y
    x_τxy = x
    y_τxy = y

    # set density & viscosity
    ρ_vy = zeros(nx + 1, ny) # on Vy-nodes
    μ_b = zeros(nx, ny) # on τxy-nodes
    μ_p = zeros(nx - 1, ny - 1) # on P-nodes (also τxx-, τyy-nodes)
    set_properties!(x_p, y_p, μ_p, plume_x, plume_y, plume_r, μ_matrix, μ_plume, μ_air, air_height) # viscosity: P-nodes
    set_properties!(x_τxy, y_τxy, μ_b, plume_x, plume_y, plume_r, μ_matrix, μ_plume, μ_air, air_height) # viscosity: τxy-nodes
    set_properties!(x_vy, y_vy, ρ_vy, plume_x, plume_y, plume_r, ρ_matrix, ρ_plume, ρ_air, air_height) # density: Vy-nodes
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
    ncheck = min(5 * min(nx, ny), 2000)
    ndtupdate = 10 * min(nx, ny)
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
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

        # Free slip BC
        @parallel (1:size(Vx, 1)) bc_y_noflux!(Vx)
        @parallel (1:size(Vx, 2)) bc_x_zero!(Vx)
        @parallel (1:size(Vy, 1)) bc_y_zero!(Vy)
        @parallel (1:size(Vy, 2)) bc_x_noflux!(Vy)

        if use_free_surface_stabilization && (iter % ndtupdate == 0)
            dt = compute_dt(Vx, Vy, maxdisp, dx, dy)
        end

        if iter % ncheck == 0
            dt_check = if use_free_surface_stabilization
                compute_dt(Vx, Vy, maxdisp, dx, dy)
            else
                0.0
            end
            @parallel compute_Residuals!(Rx, Ry, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, _dx, _dy, dt_check)
            mean_Rx = mean(abs.(Rx))
            mean_Ry = mean(abs.(Ry))
            mean_∇V = mean(abs.(∇V))
            err = maximum([mean_Rx, mean_Ry, mean_∇V])
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

    return Array(Vx), Array(Vy)
end

# this function is very slow and could be improved significantly, but is not often called
function compute_dt(Vx, Vy, maxdisp, dx, dy)
    return maxdisp * min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)))
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

@parallel function compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
    @inn(Vx) = @inn(Vx) + @all(dτVx) * @all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy) * @all(dVydτ)
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

@testset "StokesPrototype_perf" begin
    Vx, Vy = Stokes2D() # with Nx = Ny = 127
    indsx = [3, 56, 60, 90, 99]
    indsy = [28, 68, 95, 96, 127]
    refsX = [-0.0003712068980752795 0.00010905169496392347 0.000267755582008052 0.00026854265557366223 0.00024606861805851133; -0.0027482306842478597 0.0007423399653102009 0.0020363045844155043 0.0019787430110789076 0.0011279676020830934; -0.0014453266581855562 0.0003791581115906984 0.0010878875311024295 0.001052679696046908 0.0005785490645806616; 0.005199539262094441 -0.0017839561342308688 -0.0037444305230983576 -0.0036989465315485524 -0.0026371392578164645; 0.004590346143218592 -0.0014248997889780776 -0.0032836686064351846 -0.0032675668807606944 -0.002598368199775616]
    refsY = [0.004548280829338265 0.007022957782322904 0.004128436314717764 0.003994224862312994 0.0; -0.005831248664963183 -0.010571252916218401 -0.0050169200768207375 -0.004809406556908581 0.0; -0.006626577435333647 -0.011676634743429566 -0.005642539667997879 -0.005397750069403199 0.0; -0.0005179329596351789 0.0010674231624409325 -0.0006137835671708514 -0.0006220625145322597 0.0; 0.0017833460682873507 0.003728956033699204 0.001445367110859282 0.0013714620231310677 0.0]
    @test all(isapprox.(refsX, Vx[indsx, indsy]; atol=3e-7))
    @test all(isapprox.(refsY, Vy[indsx, indsy]; atol=3e-7))
end
