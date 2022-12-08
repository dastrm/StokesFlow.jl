#=
This script solves the 2D Stokes & Continuity equations for given arrays of μ, ρ.

∂τxx/∂x + ∂τxy/∂y - ∂P/∂x = 0
∂τyx/∂x + ∂τyy/∂y - ∂P/∂y = -ρ*g*y
∂vx/∂x + ∂vy/∂y = 0
τxx = 2μ*∂vx/∂x
τyy = 2μ*∂vy/∂y
τxy = τyx = μ*(∂vx/∂y + ∂vy/∂x)

PT iterations are implemented as in the 2D minapp of ParallelStencil.jl
https://github.com/omlins/ParallelStencil.jl/blob/main/miniapps/Stokes2D.jl
=#

#=
What's still missing:
- Free slip BC and correct array sizes for BC implementation
- Performance (Kernel fusing, tuning relaxation parameters, ...)
- Adapting the equations to avoid extremely large visosities, maybe pressures
- Also use viscosity on τxy-nodes, i.e. μ_τxy instead of averaging of pressure nodes μ_p
=#

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics


@views function Stokes2D()
    # Physics
    lx, ly = 100000, 100000 # domain size
    μ_air, μ_matrix, μ_plume = 1e17, 1e19, 1e18 # Viscosity TODO units
    ρ_air, ρ_matrix, ρ_plume = 1   , 3300, 3200 # Density TODO units
    g_y = 9.81
    plume_x, plume_y = lx/2, ly/2 # plume midpoint
    plume_r = 20000 # plume radius
    air_height = 0.2*ly

    #μ_air = μ_matrix
    #ρ_air = ρ_matrix

    # Numerics
    nx, ny = 127,127#35, 45

    # Derived Numerics
    dx, dy = lx/(nx-1), ly/(ny-1)

    # Array allocations
    P    = @zeros(nx  ,ny  )
    τxx  = @zeros(nx  ,ny  )
    τyy  = @zeros(nx  ,ny  )
    Vx   = @zeros(nx+1,ny  )
    Vy   = @zeros(nx  ,ny+1)
    τxy  = @zeros(nx-1,ny-1)

    # coordinates for points
    x_p  = [(ix-1)*dx       for ix=1:nx  ]
    y_p  = [(iy-1)*dy       for iy=1:ny  ]
    x_vy = [(ix-1)*dx       for ix=1:nx  ]
    y_vy = [(iy-1)*dy-0.5dy for iy=1:ny+1]
    x_τxy= [(ix-1)*dx+0.5dx for ix=1:nx-1]
    y_τxy= [(iy-1)*dy+0.5dy for iy=1:ny-1]

    # set density & viscosity
    ρ_vy  = zeros(nx  ,ny+1) # on Vy-nodes
    μ_τxy = zeros(nx-1,ny-1) # on τxy-nodes
    μ_p   = zeros(nx  ,ny  ) # on P-nodes (also τxx-, τyy-nodes)
    set_properties!(x_p  ,y_p  ,μ_p  ,plume_x,plume_y,plume_r,μ_matrix,μ_plume,μ_air,air_height) # viscosity: P-nodes
    set_properties!(x_τxy,y_τxy,μ_τxy,plume_x,plume_y,plume_r,μ_matrix,μ_plume,μ_air,air_height) # viscosity: τxy-nodes
    set_properties!(x_vy ,y_vy ,ρ_vy ,plume_x,plume_y,plume_r,ρ_matrix,ρ_plume,ρ_air,air_height) # density: Vy-nodes
    ρ_vy  = Data.Array(ρ_vy)
    μ_τxy = Data.Array(μ_τxy)
    μ_p   = Data.Array(μ_p)

    # plot density & viscosity
    p1 = heatmap(x_p  ,y_p  ,Array(μ_p)'  ,yflip=true,title="Viscosity μ_p"  ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p2 = heatmap(x_τxy,y_τxy,Array(μ_τxy)',yflip=true,title="Viscosity μ_τxy",xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p3 = heatmap(x_vy ,y_vy ,Array(ρ_vy)' ,yflip=true,title="Density ρ_vy"   ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    display(plot(p1,p2,p3))
    #display(plot(p1)); display(plot(p2)); display(plot(p3))


    # more arrays
    ∇V        = @zeros(nx  ,ny  )
    dτPt      = @zeros(nx  ,ny  )
    Rx        = @zeros(nx-1,ny-2)
    Ry        = @zeros(nx-2,ny-1)
    dVxdτ     = @zeros(nx-1,ny-2)
    dVydτ     = @zeros(nx-2,ny-1)
    dτVx      = @zeros(nx-1,ny-2)
    dτVy      = @zeros(nx-2,ny-1)
    Rog = ρ_vy[:,2:end]*g_y # hack
    Vdmp      = 4.0
    Vsc       = 0.25#1.0         # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0/4.0 
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)
    dampX     = 1.0-Vdmp/nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/ny # damping term for the y-momentum equation
    

    # PT setup and iterations
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    ncheck = 20
    t1 = 0; itert1 = 11
    ϵ = 0.01 # tol
    err_evo1=[]; err_evo2=[]
    err = 2ϵ; iter=1; niter=0; iterMax=100000
    while err > ϵ && iter <= iterMax
        if (iter==itert1) t1 = Base.time() end

        # TODO use μ_τxy where appropriate
        # TODO improve kernels
        # TODO make var names consistent
        @parallel compute_P!(∇V, P, Vx, Vy, dτPt, dx, dy)
        @parallel compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, μ_p, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVxdτ, dVydτ, P, Rog, τxx, τyy, τxy, dampX, dampY, dx, dy)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

        @parallel (1:size(Vx,1), 1:size(Vx,2)) bc_y!(Vx)
        @parallel (1:size(Vy,1), 1:size(Vy,2)) bc_x!(Vy)

        if iter%ncheck == 0
            # TODO compute error better
            mean_Rx = mean(abs.(Rx)); mean_Ry = mean(abs.(Ry)); mean_∇V = mean(abs.(∇V))
            err = maximum([mean_Rx, mean_Ry, mean_∇V])
            push!(err_evo1, err); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_∇V=%1.3e] \n", iter, err, mean_Rx, mean_Ry, mean_∇V)
        end

        iter+=1; niter+=1
    end
    t2 = Base.time()
    t_it = (t2-t1)/(niter-itert1+1)
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number) 
    T_eff    = A_eff/t_it
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, t2-t1, round(T_eff, sigdigits=2))


    # Visualization
    p1 = heatmap(x_p ,  y_p, Array(P)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Pressure")
    p2 = heatmap(x_vy, y_vy, Array(Vy)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vy")
    #p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:inferno, title="log10(Ry)")
    p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    #display(plot(p1, p2, p4, p5))
    display(plot(p1,p2,p5))

    return nothing
end


@parallel function compute_timesteps!(dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, Mus::Data.Array, Vsc::Data.Number, Ptsc::Data.Number, min_dxy2::Data.Number, max_nxy::Int)
    @all(dτVx) = Vsc*min_dxy2/@av_xi(Mus)/4.1
    @all(dτVy) = Vsc*min_dxy2/@av_yi(Mus)/4.1
    @all(dτPt) = Ptsc*4.1*@all(Mus)/max_nxy
    return
end

@parallel function compute_P!(∇V::Data.Array, Pt::Data.Array, Vx::Data.Array, Vy::Data.Array, dτPt::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(∇V)  = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Vx::Data.Array, Vy::Data.Array, Mus::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(τxx) = 2.0*@all(Mus)*(@d_xa(Vx)/dx - 1.0/3.0*@all(∇V))
    @all(τyy) = 2.0*@all(Mus)*(@d_ya(Vy)/dy - 1.0/3.0*@all(∇V))
    @all(τxy) = 2.0*@av(Mus)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, Pt::Data.Array, Rog::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, dampX::Data.Number, dampY::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Rx)    = @d_xi(τxx)/dx + @d_ya(τxy)/dy - @d_xi(Pt)/dx
    @all(Ry)    = @d_yi(τyy)/dy + @d_xa(τxy)/dx - @d_yi(Pt)/dy + @av_yi(Rog)
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)
    @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    return
end

@parallel_indices (ix,iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix,iy) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end


# This function sets material properties in arr depending on where the node is
function set_properties!(x,y,arr,plume_x,plume_y,plume_r,matrix,plume,air,air_height)
    nx, ny = size(arr)
    @assert nx <= size(x,1) && ny <= size(y,1)
    for ix=1:nx
        for iy=1:ny
            val = 0.0
            if y[iy] < air_height
                val = air
            elseif (x[ix]-plume_x)^2 + (y[iy]-plume_y)^2 <= plume_r^2
                val = plume
            else
                val = matrix
            end
            arr[ix,iy] = val
        end
    end
    return nothing
end



Stokes2D()
