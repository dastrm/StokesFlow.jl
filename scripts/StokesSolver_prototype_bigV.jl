#=
This script only differs from StokesSolver_prototype.jl in the following points:

    * Vx is extended by 2 nodes in x-direction
    * Vy is extended by 2 nodes in y-direction
    * => computation does not even read these additional entries

Motivation: These added entries are needed for velocity interpolation from markers
            (coming from other ranks in case of multi-xPU,
            set such that the second derivative is 0, in case of physical boundary)

However:    Memory layout is worse for the Stokes Solver.
            This script shows ~5-10% slower performance.
            Thus, it is probably better to add this additional padding after the Stokes Solver.
=#


const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

macro within(macroname::String, A::Symbol)
    if     macroname == "@all"    esc(  :($ix<=size($A,1)   && $iy<=size($A,2)  )  )
    elseif macroname == "@inn"    esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)-2)  )
    elseif macroname == "@inn_x"  esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)  )  )
    elseif macroname == "@inn_y"  esc(  :($ix<=size($A,1)   && $iy<=size($A,2)-2)  )
    elseif macroname == "@inn_innx"  esc(  :($ix<=size($A,1)-4 && $iy<=size($A,2)-2)  )
    elseif macroname == "@inn_inny"  esc(  :($ix<=size($A,1)-2 && $iy<=size($A,2)-4)  )
    # I can't get what's mentioned in the next line to work, thus the entire macro is just copied and extended as short-term workaround
    else error("unkown macroname: $macroname. If you want to add your own assignement macros, overwrite the macro 'within(macroname::String, A::Symbol)'; to still use the exising macro within as well call ParallelStencil.FiniteDifferences{1|2|3}D.@within(macroname, A) at the end.")
    end
end

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, Test

# ADDITIONAL PARALLEL STENCIL MACROS, needed for free surface stabilization and larger Vx / Vy array sizes
import ..ParallelStencil: INDICES
ix, iy = INDICES[1], INDICES[2]
ixi, iyi = :($ix+1), :($iy+1)

macro    av_inn(A::Symbol)  esc(:(($A[$ixi  ,$iyi ] + $A[$ixi+1,$iyi ] + $A[$ixi,$iyi+1] + $A[$ixi+1,$iyi+1])*0.25 )) end # average in both dimension, and inner elements in both directions. corresponds to @av(@inn_x(@inn_y(..)))
macro    d_xi_2(A::Symbol)  esc(:( $A[$ix+2, $iyi ] - $A[$ix  , $iyi] )) end # central finite differences in x, inner elements in y. corresponds to @d_xi(@av_x(..))
macro    d_yi_2(A::Symbol)  esc(:( $A[$ixi  ,$iy+2] - $A[$ixi , $iy ] )) end # central finite differences in y, inner elements in x. corresponds to @d_yi(@av_y(..))
macro     d_xii(A::Symbol)  esc(:( $A[$ixi+1,$iyi ] - $A[$ixi  ,$iyi] )) end # differences along inner elements in x, select inner elements in y
macro     d_yii(A::Symbol)  esc(:( $A[$ixi ,$iyi+1] - $A[$ixi ,$iyi ] )) end # differences along inner elements in y, select inner elements in x
macro  inn_innx(A::Symbol)  esc(:( $A[$ixi+1 ,$iyi] )) end # inner elements n x, twice inner elements in y
macro  inn_inny(A::Symbol)  esc(:( $A[$ixi ,$iyi+1] )) end # inner elements n x, twice inner elements in y

@views function Stokes2D()
    # Physics
    lx, ly = [100000, 100000]./100000 # domain size
    μ_air, μ_matrix, μ_plume = [1e17, 1e19, 1e18]./1e19 # Viscosity
    ρ_air, ρ_matrix, ρ_plume = [1   , 3300, 3200]./1000 # Density
    g_y = 9.81
    plume_x, plume_y = lx/2, ly/2 # plume midpoint
    plume_r = 20000 / 100000 # plume radius
    air_height = 0.2*ly

    use_free_surface_stabilization = true
    maxdisp = 0.5

    # Numerics
    nx, ny = 127,127#35, 45

    # Derived Numerics
    dx, dy = lx/(nx-1), ly/(ny-1)

    # Array allocations
    P    = @zeros(nx-1,ny-1)
    τxx  = @zeros(nx-1,ny-1) #p-nodes
    τyy  = @zeros(nx-1,ny-1) #p-nodes
    Vx   = @zeros(nx+2,ny+1)
    Vy   = @zeros(nx+1,ny+2)
    τxy  = @zeros(nx  ,ny  )

    # coordinates for points
    x    = [(ix-1)*dx       for ix=1:nx  ] # basic nodes = τ_xy-nodes
    y    = [(iy-1)*dy       for iy=1:ny  ]
    x_p  = [(ix-1)*dx+0.5dx for ix=1:nx-1] # pressure nodes
    y_p  = [(iy-1)*dy+0.5dy for iy=1:ny-1]
    x_vx = x                               # Vx nodes
    y_vx = [(iy-1)*dy-0.5dy for iy=1:ny+1]
    x_vy = [(ix-1)*dx-0.5dx for ix=1:nx+1] # Vy nodes
    y_vy = y
    x_τxy= x
    y_τxy= y

    # set density & viscosity
    ρ_vy  = zeros(nx+1,ny  ) # on Vy-nodes
    μ_τxy = zeros(nx  ,ny  ) # on τxy-nodes
    μ_p   = zeros(nx-1,ny-1) # on P-nodes (also τxx-, τyy-nodes)
    set_properties!(x_p  ,y_p  ,μ_p  ,plume_x,plume_y,plume_r,μ_matrix,μ_plume,μ_air,air_height) # viscosity: P-nodes
    set_properties!(x_τxy,y_τxy,μ_τxy,plume_x,plume_y,plume_r,μ_matrix,μ_plume,μ_air,air_height) # viscosity: τxy-nodes
    set_properties!(x_vy ,y_vy ,ρ_vy ,plume_x,plume_y,plume_r,ρ_matrix,ρ_plume,ρ_air,air_height) # density: Vy-nodes
    ρ_vy  = Data.Array(ρ_vy)
    μ_τxy = Data.Array(μ_τxy)
    μ_p   = Data.Array(μ_p)
#=
    # plot density & viscosity
    p1 = heatmap(x_p  ,y_p  ,Array(μ_p)'  ,yflip=true,title="Viscosity μ_p"  ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p2 = heatmap(x_τxy,y_τxy,Array(μ_τxy)',yflip=true,title="Viscosity μ_τxy",xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    p3 = heatmap(x_vy ,y_vy ,Array(ρ_vy)' ,yflip=true,title="Density ρ_vy"   ,xlabel='x',ylabel='y',xlims=(0,lx),ylims=(0,ly),aspect_ratio=1)
    #display(plot(p1,p2,p3))
    #display(plot(p1)); display(plot(p2)); display(plot(p3))
=#

    # more arrays
    ∇V        = @zeros(nx-1,ny-1)
    dτPt      = @zeros(nx-1,ny-1)
    Rx        = @zeros(nx-2,ny-1)
    Ry        = @zeros(nx-1,ny-2)
    dVxdτ     = @zeros(nx-2,ny-1)
    dVydτ     = @zeros(nx-1,ny-2)
    dτVx      = @zeros(nx-2,ny-1)
    dτVy      = @zeros(nx-1,ny-2)
    Vdmp      = 4.0
    Vsc       = 0.25#1.0         # relaxation paramter for the momentum equations pseudo-timesteps limiters
    Ptsc      = 1.0/4.0 
    min_dxy2  = min(dx,dy)^2
    max_nxy   = max(nx,ny)
    dampX     = 1.0-Vdmp/nx # damping term for the x-momentum equation
    dampY     = 1.0-Vdmp/ny # damping term for the y-momentum equation
    
    dt = 0.0

    # PT setup and iterations
    @parallel compute_timesteps!(dτVx, dτVy, dτPt, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    ncheck = 500
    t1 = 0; itert1 = 11
    ϵ = 0.01 # tol
    err_evo1=[]; err_evo2=[]
    err = 2ϵ; iter=1; niter=0; iterMax=100000
    while err > ϵ && iter <= iterMax
        if (iter==itert1) t1 = Base.time() end

        # TODO improve kernels
        # TODO make var names consistent
        @parallel compute_P!(∇V, P, Vx, Vy, dτPt, dx, dy)
        @parallel compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, μ_p, μ_τxy, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVxdτ, dVydτ, P, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, dx, dy, dt)
        @parallel compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

        # Free slip BC
        @parallel (1:size(Vx,1)) bc_y_noflux!(Vx)
        @parallel (1:size(Vx,2)) bc_x_zero!(Vx)
        @parallel (1:size(Vy,1)) bc_y_zero!(Vy)
        @parallel (1:size(Vy,2)) bc_x_noflux!(Vy)

        if use_free_surface_stabilization
            dt = maxdisp*min(dx/maximum(Vx),dy/maximum(Vy))
        end

        if iter%ncheck == 0
            # TODO compute error better
            mean_Rx = mean(abs.(Rx)); mean_Ry = mean(abs.(Ry)); mean_∇V = mean(abs.(∇V))
            err = maximum([mean_Rx, mean_Ry, mean_∇V])
            push!(err_evo1, err); push!(err_evo2,iter)
            #@printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_∇V=%1.3e] \n", iter, err, mean_Rx, mean_Ry, mean_∇V)
        end

        iter+=1; niter+=1
    end
    t2 = Base.time()
    t_it = (t2-t1)/(niter-itert1+1)
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number) 
    T_eff    = A_eff/t_it
    @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ T_eff = %1.2f GB/s) \n", niter, err, t2-t1, round(T_eff, sigdigits=2))

#=
    # Visualization
    p1 = heatmap(x_p ,  y_p, Array(P)' , yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Pressure")
    p2 = heatmap(x_vy, y_vy, Array(Vy)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vy")
    #p4 = heatmap(X[2:end-1], Yv[2:end-1], log10.(abs.(Array(Ry)')), aspect_ratio=1, xlims=(X[2],X[end-1]), ylims=(Yv[2],Yv[end-1]), c=:inferno, title="log10(Ry)")
    p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    p6 = heatmap(x_vx, y_vx, Array(Vx)', yflip=true, aspect_ratio=1, xlims=(0,lx), ylims=(0,ly), c=:inferno, title="Vx")
    #display(plot(p1, p2, p4, p5))
    #display(plot(p1,p2,p5,p6))
=#

    return Array(Vx), Array(Vy)
end


@parallel function compute_timesteps!(dτVx, dτVy, dτPt, μ_p, Vsc, Ptsc, min_dxy2, max_nxy)
    @all(dτVx) = Vsc*min_dxy2/@av_xa(μ_p)/4.1
    @all(dτVy) = Vsc*min_dxy2/@av_ya(μ_p)/4.1
    @all(dτPt) = Ptsc*4.1*@all(μ_p)/max_nxy
    return
end

@parallel function compute_P!(∇V, Pt, Vx, Vy, dτPt, dx, dy)
    @all(∇V)  = @d_xii(Vx)/dx + @d_yii(Vy)/dy
    @all(Pt)  = @all(Pt) - @all(dτPt)*@all(∇V)
    return
end

@parallel function compute_τ!(∇V, τxx, τyy, τxy, Vx, Vy, μ_p, μ_τxy, dx, dy)
    @all(τxx) = 2.0*@all(μ_p)*(@d_xii(Vx)/dx - 1.0/3.0*@all(∇V))
    @all(τyy) = 2.0*@all(μ_p)*(@d_yii(Vy)/dy - 1.0/3.0*@all(∇V))
    @all(τxy) = 2.0*@all(μ_τxy)*(0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx))
    return
end

@parallel function compute_dV!(Rx, Ry, dVxdτ, dVydτ, Pt, ρ_vy, τxx, τyy, τxy, Vx, Vy, g_y, dampX, dampY, dx, dy, dt)
    @all(Rx)    = @d_xa(τxx)/dx + @d_yi(τxy)/dy - @d_xa(Pt)/dx
    @all(Ry)    = @d_ya(τyy)/dy + @d_xi(τxy)/dx - @d_ya(Pt)/dy + 
                    g_y*(@inn(ρ_vy) - dt*(@av_inn(Vx)*@d_xi_2(ρ_vy)/(2*dx) + @inn_inny(Vy)*@d_yi_2(ρ_vy)/(2*dy)))
    @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)
    return
end

@parallel function compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)
    @inn_innx(Vx) = @inn_innx(Vx) + @all(dτVx)*@all(dVxdτ)
    @inn_inny(Vy) = @inn_inny(Vy) + @all(dτVy)*@all(dVydτ)
    return
end

# side boundaries
@parallel_indices (iy) function bc_x_zero!(A::Data.Array)
    A[1    , iy] = 0.0 # actually not necessary, must only be done once
    A[2    , iy] = 0.0
    A[end-1, iy] = 0.0
    A[end  , iy] = 0.0 # actually not necessary, must only be done once
    return
end
@parallel_indices (iy) function bc_x_noflux!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

# horizontal boundaries
@parallel_indices (ix) function bc_y_zero!(A::Data.Array)
    A[ix, 1    ] = 0.0 # actually not necessary, must only be done once
    A[ix, 2    ] = 0.0
    A[ix, end-1] = 0.0
    A[ix, end  ] = 0.0 # actually not necessary, must only be done once
    return
end
@parallel_indices (ix) function bc_y_noflux!(A::Data.Array)
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

#Stokes2D()

@testset "StokesPrototypeBigV" begin
    Vx, Vy = Stokes2D()
    indsx = [3, 56, 60, 90, 99]
    indsy = [28, 68, 95, 96, 127]
    refsX = [-0.0004256954586403605 1.5704371157731378e-5 0.00020317457002628982 0.00020485661003272825 0.00019807580217962747; -0.002954374891725917 6.605275884106962e-5 0.0015886734760139 0.001549589156472289 0.0009164422446087828; -0.001544791481653481 3.530760363148521e-5 0.0008506677047454428 0.000825938977338414 0.0004701929646878827; 0.005828514660768712 -9.509562075414608e-5 -0.002886834816671759 -0.0028648080397676227 -0.002135766923882878; 0.005211592848221708 -0.0001942009430386829 -0.002519056628485078 -0.002518564909688592 -0.0020998610702983845]
    refsY = [0.00018118060009758292 0.005440240490767683 0.0036911883776263987 0.003582582196461027 0.0; -0.007326166456758924 -0.010428909007936273 -0.004082459048205107 -0.0039106270522065445 0.0; -0.007876334552195765 -0.011431292694232263 -0.0046128232709835085 -0.004409907104060302 0.0; -0.003475950534627877 0.0002256859223269555 -0.00035750322051714324 -0.00036212056844438597 0.0; -0.0017734202974535158 0.0026446919182487416 0.0013945745718235373 0.0013360116402855989 0.0]
    @test all(refsX .≈ Vx[indsx.+1,indsy])
    @test all(refsY .≈ Vy[indsx,indsy.+1])
end
