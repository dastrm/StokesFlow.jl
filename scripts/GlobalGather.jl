using ImplicitGlobalGrid, MPI

"""
This function gathers all markers (having local coordinates) into arrays with global coordinates.
Result is only valid on rank 0.
"""
function gather_markers!(x_m,y_m,ρ_m,μ_m,x_m_glob,y_m_glob,ρ_m_glob,μ_m_glob,dims,dx,dy,lx_loc,ly_loc,me,comm_cart)
    Nm_loc = length(x_m)
    @assert Nm_loc == length(y_m)
    num_markers = MPI.Allgather(Int32(Nm_loc),comm_cart)

    if me == 0
        Nm_glob = sum(num_markers)
        resize!(x_m_glob,Nm_glob)
        resize!(y_m_glob,Nm_glob)
        resize!(ρ_m_glob,Nm_glob)
        resize!(μ_m_glob,Nm_glob)
    end

    if me==0
        MPI.Gatherv!(x_m,MPI.VBuffer(x_m_glob,num_markers),0,comm_cart)
        MPI.Gatherv!(y_m,MPI.VBuffer(y_m_glob,num_markers),0,comm_cart)
        MPI.Gatherv!(ρ_m,MPI.VBuffer(ρ_m_glob,num_markers),0,comm_cart)
        MPI.Gatherv!(μ_m,MPI.VBuffer(μ_m_glob,num_markers),0,comm_cart)
    else
        MPI.Gatherv!(x_m,nothing,0,comm_cart)
        MPI.Gatherv!(y_m,nothing,0,comm_cart)
        MPI.Gatherv!(ρ_m,nothing,0,comm_cart)
        MPI.Gatherv!(μ_m,nothing,0,comm_cart)
    end
    
    # shift marker coords
    if me == 0
        for xproc in 0:dims[1]-1
            for yproc in 0:dims[2]-1
                rank = MPI.Cart_rank(comm_cart,[xproc,yproc,0])
                glob_m_first = sum(num_markers[1:rank]) + 1
                glob_m_last = glob_m_first + num_markers[rank+1] - 1
                x_shift = xproc*(lx_loc-dx)
                y_shift = yproc*(ly_loc-dy)
                x_m_glob[glob_m_first:glob_m_last] .+= x_shift
                y_m_glob[glob_m_first:glob_m_last] .+= y_shift
            end
        end
    end

    return
end

"""
This function gathers local grid arrays of Vx and Vy into arrays that look as if only a single process was used.
Result is only valid on rank 0.
"""
function gather_V_grid(Vx_loc, Vy_loc, me, dims, nx, ny)
    Vx_glob = if me == 0 zeros(Tuple(dims[1:2]).*size(Vx_loc)) else zeros(0,0) end
    Vy_glob = if me == 0 zeros(Tuple(dims[1:2]).*size(Vy_loc)) else zeros(0,0) end
    gather!(Array(Vx_loc),Vx_glob; root=0)
    gather!(Array(Vy_loc),Vy_glob; root=0)
    if me == 0
        Vx_glob = create_grid_view(Vx_glob,dims,nx,ny,size(Vx_loc,1),size(Vx_loc,2))
        Vy_glob = create_grid_view(Vy_glob,dims,nx,ny,size(Vy_loc,1),size(Vy_loc,2))
    end
    return Vx_glob, Vy_glob
end


"""
This function creates global grid arrays that look as if only a single process was used from
a global grid array containing ALL entries from all ranks, i.e. the result from a call to gather!(..) from ImplicitGlobalGrid.
E.g. see gather_V_grid(..) for usage.
"""
function create_grid_view(A_all,dims,Nx_loc,Ny_loc,Nx_A,Ny_A)
    @assert dims[3] == 1
    A_x,A_y = size(A_all)
    x_inds = 1:A_x
    y_inds = 1:A_y
    # compute overlap
    o_x = 2+Nx_A-Nx_loc
    o_y = 2+Ny_A-Ny_loc
    # remove overlapping indices
    for xproc in 1:dims[1]-1
        nl = o_x÷2
        nr = nl + o_x%2
        c = xproc*Nx_A
        rem = c+1-nl:c+nr
        x_inds = setdiff(x_inds,rem)
    end
    for yproc in 1:dims[2]-1
        nl = o_y÷2
        nr = nl + o_y%2
        c = yproc*Ny_A
        rem = c+1-nl:c+nr
        y_inds = setdiff(y_inds,rem)
    end
    v = @view A_all[x_inds,y_inds]
    vsize = size(v)
    @assert vsize[1] == nx_g()+o_x-2 && vsize[2] == ny_g()+o_y-2 
    return v
end