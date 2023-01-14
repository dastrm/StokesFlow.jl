@assert (@isdefined(USE_GPU)) "USE_GPU must be defined"
typeassert(USE_GPU, Bool)

using ParallelStencil, ImplicitGlobalGrid
import MPI

@static if USE_GPU
    import CUDA
end

include("common.jl")

"""
    bilinearMarkerToGrid!(x_grid_min, y_grid_min, val_grid, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum, grid::::ImplicitGlobalGrid.GlobalGrid)

Interpolates markers to grid points
"""
@views function bilinearMarkerToGrid!(x_grid_min, y_grid_min, val_grid, x_m, y_m, val_m, dx, dy, val_wt_sum, wt_sum, grid::ImplicitGlobalGrid.GlobalGrid)

    Nx, Ny = size(val_grid)
    Nm = size(x_m, 1)
    @assert (size(y_m, 1) == Nm) && (size(val_m, 1) == Nm)

    val_wt_sum .= 0.0
    wt_sum .= 0.0

    # interpolate by summing up the marker contributions to global arrays
    sum_up_local_contributions!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy, Nm)

    # perform reduction on boundaries
    sumOverlappingValues!(wt_sum, grid)
    sumOverlappingValues!(val_wt_sum, grid)

    # finally compute actual value from the sums
    @parallel (1:Nx, 1:Ny) safeDivision(val_grid, val_wt_sum, wt_sum)

    return nothing
end

@static if USE_GPU
    @inline function sum_up_local_contributions!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy, Nm)
        @parallel (1:Nm) atomicAddInterpolationCUDA!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy)
        return nothing
    end
else
    @inline function sum_up_local_contributions!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy, Nm)
        singleCoreInterpolationCPU!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy, Nm)
        return nothing
    end
end

@static if USE_GPU
    """
        atomicAddInterpolationCUDA!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy)

    Sums up all interpolation weights in an atomic fashion
    """
    @parallel_indices (m) function atomicAddInterpolationCUDA!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy)
        xm = x_m[m]
        ym = y_m[m]
        val = val_m[m]

        # get indices and relative distance to top left node w.r.t marker m.
        # may be 0, when the marker is further left or up than the first grid node
        ix, iy, dxmij, dymij = topleftIndexRelDist(x_grid_min, y_grid_min, xm, ym, dx, dy)
        if !(ix >= 0 && ix <= Nx && iy >= 0 && iy <= Ny)
            @printline("ATTENTION: erroneous marker position")
        else
            # sum up weights, if the respective node exists
            if iy > 0
                if ix > 0
                    # 1) top left
                    w = (1 - dxmij) * (1 - dymij)
                    CUDA.@atomic wt_sum[ix, iy] += w
                    CUDA.@atomic val_wt_sum[ix, iy] += w * val
                end
                if ix < Nx
                    # 2) top right
                    w = dxmij * (1 - dymij)
                    CUDA.@atomic wt_sum[ix+1, iy] += w
                    CUDA.@atomic val_wt_sum[ix+1, iy] += w * val
                end
            end
            if iy < Ny
                if ix > 0
                    # 3) bottom left
                    w = (1 - dxmij) * dymij
                    CUDA.@atomic wt_sum[ix, iy+1] += w
                    CUDA.@atomic val_wt_sum[ix, iy+1] += w * val
                end
                if ix < Nx
                    # 4) bottom right
                    w = dxmij * dymij
                    CUDA.@atomic wt_sum[ix+1, iy+1] += w
                    CUDA.@atomic val_wt_sum[ix+1, iy+1] += w * val
                end
            end
        end
        return nothing
    end
end


@static if !USE_GPU
    """
        singleCoreInterpolationCPU!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy, Nm)

    Sums up all interpolation weights using arrays
    """
    @inline function singleCoreInterpolationCPU!(x_m, y_m, val_m, wt_sum, val_wt_sum, Nx, Ny, x_grid_min, y_grid_min, dx, dy, Nm)
        for m = 1:1:Nm

            xm = x_m[m]
            ym = y_m[m]
            val = val_m[m]

            # get indices and relative distance to top left node w.r.t marker m.
            # may be 0, when the marker is further left or up than the first grid node
            ix, iy, dxmij, dymij = topleftIndexRelDist(x_grid_min, y_grid_min, xm, ym, dx, dy)
            if !(ix >= 0 && ix <= Nx && iy >= 0 && iy <= Ny)
                @printline("ATTENTION: erroneous marker position")
            else
                # sum up weights, if the respective node exists
                if iy > 0
                    if ix > 0
                        # 1) top left
                        w = (1 - dxmij) * (1 - dymij)
                        wt_sum[ix, iy] += w
                        val_wt_sum[ix, iy] += w * val
                    end
                    if ix < Nx
                        # 2) top right
                        w = dxmij * (1 - dymij)
                        wt_sum[ix+1, iy] += w
                        val_wt_sum[ix+1, iy] += w * val
                    end
                end
                if iy < Ny
                    if ix > 0
                        # 3) bottom left
                        w = (1 - dxmij) * dymij
                        wt_sum[ix, iy+1] += w
                        val_wt_sum[ix, iy+1] += w * val
                    end
                    if ix < Nx
                        # 4) bottom right
                        w = dxmij * dymij
                        wt_sum[ix+1, iy+1] += w
                        val_wt_sum[ix+1, iy+1] += w * val
                    end
                end
            end
        end
        return
    end
end


"""
    safeDivision(result, numerator, denominator)

Performs division and warns about zero denominator
"""
@parallel_indices (ix, iy) function safeDivision(result, numerator, denominator)
    denom = denominator[ix, iy]
    if denom != 0.0
        result[ix, iy] = numerator[ix, iy] / denom
    else
        @printline("PROBLEM: no markers close enough for value interpolation at node: ix = ", ix, ", iy = ", iy)
    end
    return nothing
end

"""
    sumOverlappingValues!(A::Data.Array, grid::ImplicitGlobalGrid.GlobalGrid)

Sums up all values of the local array A that are overlapping in the global grid.
Results are stored back in the corresponding entries of A.

Implementation is not particularly efficient (especially memory-wise),
but enough since it is not called very often.
"""
@views function sumOverlappingValues!(A::Data.Array, grid::ImplicitGlobalGrid.GlobalGrid)

    nb = grid.neighbors
    comm_cart = grid.comm

    ox = grid.overlaps[1] + size(A, 1) - grid.nxyz[1]
    oy = grid.overlaps[2] + size(A, 2) - grid.nxyz[2]

    @assert 2 * ox <= size(A, 1) && 2 * oy <= size(A, 2) "Total overlap is bigger than Array itself"

    # neighbors
    below = nb[2, 2]
    above = nb[1, 2]
    right = nb[2, 1]
    left = nb[1, 1]

    # compute the sums: first in x, then in y. must be separate for correct results in corners.
    # exchange in x-direction
    if ox > 0
        # receive buffers
        recv_buf_x_lr = zeros(ox, size(A, 2))
        recv_buf_x_rl = zeros(ox, size(A, 2))
        # prepare requests
        reqs = fill(MPI.REQUEST_NULL, 4)
        # receive
        if (left != MPI.MPI_PROC_NULL)
            reqs[1] = MPI.Irecv!(recv_buf_x_lr, left, 13, comm_cart)
        end
        if (right != MPI.MPI_PROC_NULL)
            reqs[2] = MPI.Irecv!(recv_buf_x_rl, right, 23, comm_cart)
        end
        # send
        if (right != MPI.MPI_PROC_NULL)
            reqs[3] = MPI.Isend(Array(A[end-ox+1:end, :]), right, 13, comm_cart)
        end
        if (left != MPI.MPI_PROC_NULL)
            reqs[4] = MPI.Isend(Array(A[1:ox, :]), left, 23, comm_cart)
        end
        # wait
        if any(reqs .!= [MPI.REQUEST_NULL])
            MPI.Waitall!(reqs)
        end
        # update
        if (left != MPI.MPI_PROC_NULL)
            A[1:ox, :] .+= Data.Array(recv_buf_x_lr)
        end
        if (right != MPI.MPI_PROC_NULL)
            A[end-ox+1:end, :] .+= Data.Array(recv_buf_x_rl)
        end
    end

    # exchange in y-direction
    if oy > 0
        # receive buffers
        recv_buf_y_down = zeros(size(A, 1), oy)
        recv_buf_y_up = zeros(size(A, 1), oy)
        # prepare requests
        reqs = fill(MPI.REQUEST_NULL, 4)
        # receive
        if (above != MPI.MPI_PROC_NULL)
            reqs[1] = MPI.Irecv!(recv_buf_y_down, above, 7, comm_cart)
        end
        if (below != MPI.MPI_PROC_NULL)
            reqs[2] = MPI.Irecv!(recv_buf_y_up, below, 17, comm_cart)
        end
        #send
        if (below != MPI.MPI_PROC_NULL)
            reqs[3] = MPI.Isend(Array(A[:, end-oy+1:end]), below, 7, comm_cart)
        end
        if (above != MPI.MPI_PROC_NULL)
            reqs[4] = MPI.Isend(Array(A[:, 1:oy]), above, 17, comm_cart)
        end
        # wait
        if any(reqs .!= [MPI.REQUEST_NULL])
            MPI.Waitall!(reqs)
        end
        # update
        if (above != MPI.MPI_PROC_NULL)
            A[:, 1:oy] .+= Data.Array(recv_buf_y_down)
        end
        if (below != MPI.MPI_PROC_NULL)
            A[:, end-oy+1:end] .+= Data.Array(recv_buf_y_up)
        end
    end

    return nothing
end
