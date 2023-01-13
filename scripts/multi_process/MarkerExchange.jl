import MPI

"""
    exchangeMarkers!(comm, dims, localDomain, dx, dy, x_m, y_m, ρ_m, μ_m)

Exchanges markers outside of local boundaries to respective neighbouring ranks
"""
@views function exchangeMarkers!(comm, dims, localDomain, dx, dy, x_m, y_m, ρ_m, μ_m)
    numNeighbors = 8
    coords = MPI.Cart_coords(comm)
    # local order of neighbour indices for rank x is given as follows:
    # 7       6       5
    #     ---------    
    #     |       |    
    # 8   |   x   |   4
    #     |       |    
    #     ---------    
    # 1       2       3

    # calculate send / receive ranks
    dstRanks = zeros(Int32, numNeighbors)
    srcRanks = zeros(Int32, numNeighbors)
    for i = 1:numNeighbors
        ncoords = coordsToNeighbourCoords!(coords, i)
        nrank = MPI.MPI_PROC_NULL
        if (ncoords[1] >= 0) && (ncoords[1] <= dims[1] - 1) && (ncoords[2] >= 0) && (ncoords[2] <= dims[2] - 1)
            nrank = MPI.Cart_rank(comm, ncoords)
        end
        dstRanks[i] = nrank
    end
    for i = 1:numNeighbors
        srcRanks[i] = dstRanks[dstToSrcIdx!(i)]
    end

    # initialize send / receive buffers
    sendBuffersx = Array{Float64}[]
    sendBuffersy = Array{Float64}[]
    sendBuffersρ = Array{Float64}[]
    sendBuffersμ = Array{Float64}[]
    recvBuffersx = Array{Float64}[]
    recvBuffersy = Array{Float64}[]
    recvBuffersρ = Array{Float64}[]
    recvBuffersμ = Array{Float64}[]
    for _ = 1:numNeighbors
        push!(sendBuffersx, Array{Float64}(undef, 0))
        push!(sendBuffersy, Array{Float64}(undef, 0))
        push!(sendBuffersρ, Array{Float64}(undef, 0))
        push!(sendBuffersμ, Array{Float64}(undef, 0))
        push!(recvBuffersx, Array{Float64}(undef, 0))
        push!(recvBuffersy, Array{Float64}(undef, 0))
        push!(recvBuffersρ, Array{Float64}(undef, 0))
        push!(recvBuffersμ, Array{Float64}(undef, 0))
    end

    # initialize send / receive requests
    sendSizeReqs = Array{MPI.Request}(undef, 0)
    sendDataReqs = Array{MPI.Request}(undef, 0)
    recvSizeReqs = Array{MPI.Request}(undef, 0)
    recvDataReqs = Array{MPI.Request}(undef, 0)

    # initialize temporary arrays for new local markers
    tmpx = Array{Float64}(undef, 0)
    tmpy = Array{Float64}(undef, 0)
    tmpρ = Array{Float64}(undef, 0)
    tmpμ = Array{Float64}(undef, 0)

    # separate markers within the local boundary from the ones outside
    sz = size(x_m, 1)
    @assert size(y_m, 1) == sz && size(ρ_m, 1) == sz && size(μ_m, 1) == sz
    for i = 1:sz
        # dx / 2, dy / 2 to shift *half* a cell earlier (except at physical boundaries)
        dstId = posToNeighbourIdx!(localDomain, coords, dims, x_m[i], y_m[i], dx / 2, dy / 2)
        if dstId == 0
            # inside, push to tmp
            push!(tmpx, x_m[i])
            push!(tmpy, y_m[i])
            push!(tmpρ, ρ_m[i])
            push!(tmpμ, μ_m[i])
        else
            # outside, prepare to send
            shift = neighbourIdxToShift!(dstId)
            x_m[i] -= (localDomain[1] - dx) * shift[1]
            y_m[i] -= (localDomain[2] - dy) * shift[2]
            push!(sendBuffersx[dstId], x_m[i])
            push!(sendBuffersy[dstId], y_m[i])
            push!(sendBuffersρ[dstId], ρ_m[i])
            push!(sendBuffersμ[dstId], μ_m[i])
        end
    end

    # send size and buffers to destination ranks
    for i = 1:numNeighbors
        dst = dstRanks[i]
        (dst == MPI.MPI_PROC_NULL) && continue
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersx[i], 1), dst, i, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersx[i], dst, i, comm))
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersy[i], 1), dst, i + 1 * numNeighbors, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersy[i], dst, i + 1 * numNeighbors, comm))
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersρ[i], 1), dst, i + 2 * numNeighbors, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersρ[i], dst, i + 2 * numNeighbors, comm))
        push!(sendSizeReqs, MPI.Isend(size(sendBuffersμ[i], 1), dst, i + 3 * numNeighbors, comm))
        push!(sendDataReqs, MPI.Isend(sendBuffersμ[i], dst, i + 3 * numNeighbors, comm))
    end

    # initialize receive sizes buffers
    recvSizesx = Array{Int64}[]
    recvSizesy = Array{Int64}[]
    recvSizesρ = Array{Int64}[]
    recvSizesμ = Array{Int64}[]
    for _ = 1:numNeighbors
        push!(recvSizesx, zeros(Int64, 1))
        push!(recvSizesy, zeros(Int64, 1))
        push!(recvSizesρ, zeros(Int64, 1))
        push!(recvSizesμ, zeros(Int64, 1))
    end

    # receive sizes 
    for i = 1:numNeighbors
        src = srcRanks[i]
        (src == MPI.MPI_PROC_NULL) && continue
        push!(recvSizeReqs, MPI.Irecv!(recvSizesx[i], src, i, comm))
        push!(recvSizeReqs, MPI.Irecv!(recvSizesy[i], src, i + 1 * numNeighbors, comm))
        push!(recvSizeReqs, MPI.Irecv!(recvSizesρ[i], src, i + 2 * numNeighbors, comm))
        push!(recvSizeReqs, MPI.Irecv!(recvSizesμ[i], src, i + 3 * numNeighbors, comm))
    end

    # wait for all sizes to arrive
    MPI.Waitall!(recvSizeReqs)

    # receive data
    for i = 1:numNeighbors
        src = srcRanks[i]
        (src == MPI.MPI_PROC_NULL) && continue

        recvBuffersx[i] = zeros(recvSizesx[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersx[i], src, i, comm))

        recvBuffersy[i] = zeros(recvSizesy[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersy[i], src, i + 1 * numNeighbors, comm))

        recvBuffersρ[i] = zeros(recvSizesρ[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersρ[i], src, i + 2 * numNeighbors, comm))

        recvBuffersμ[i] = zeros(recvSizesμ[i][1])
        push!(recvDataReqs, MPI.Irecv!(recvBuffersμ[i], src, i + 3 * numNeighbors, comm))
    end

    # wait for the rest to complete
    MPI.Waitall!(recvDataReqs)
    MPI.Waitall!(sendSizeReqs)
    MPI.Waitall!(sendDataReqs)

    # add received markers to tmp arrays
    for i = 1:numNeighbors
        tmpx = append!(tmpx, recvBuffersx[i])
        tmpy = append!(tmpy, recvBuffersy[i])
        tmpρ = append!(tmpρ, recvBuffersρ[i])
        tmpμ = append!(tmpμ, recvBuffersμ[i])
    end

    # return tmp arrays
    return tmpx, tmpy, tmpρ, tmpμ
end

"""
    coordsToNeighbourCoords!(coords, i)

Transforms coordinates of the current rank to the neighbour coordinates for a given neighbour index
"""
@views function coordsToNeighbourCoords!(coords, i)
    @assert i >= 1 && i <= 8
    return coords .+ neighbourIdxToShift!(i)
end

"""
    neighbourIdxToShift!(i)

Calculates three-dimensional shift for a given neighbour index
"""
@views function neighbourIdxToShift!(i)
    @assert i >= 1 && i <= 8
    (i == 1) && return [-1, -1, 0]
    (i == 2) && return [0, -1, 0]
    (i == 3) && return [1, -1, 0]
    (i == 4) && return [1, 0, 0]
    (i == 5) && return [1, 1, 0]
    (i == 6) && return [0, 1, 0]
    (i == 7) && return [-1, 1, 0]
    (i == 8) && return [-1, 0, 0]
end

"""
    dstToSrcIdx!(i)

Transforms destination to source rank index
"""
@views function dstToSrcIdx!(i)
    @assert i >= 1 && i <= 8
    (i >= 5) && return (i + 4) % 9 + 1
    return i + 4
end

"""
    posToNeighbourIdx!(localDomain, coords, dims, x, y, dx, dy)

Retrieves a potential neighbour index given a local position
"""
@views function posToNeighbourIdx!(localDomain, coords, dims, x, y, dx, dy)
    if y < dy && (coords[2] != 0)
        (x < dx) && (coords[1] != 0) && return 1
        (x > localDomain[1] - dx) && (coords[1] != dims[1] - 1) && return 3
        return 2
    end
    if y > localDomain[2] - dy && (coords[2] != dims[2] - 1)
        (x < dx) && (coords[1] != 0) && return 7
        (x > localDomain[1] - dx) && (coords[1] != dims[1] - 1) && return 5
        return 6
    end
    (x < dx) && (coords[1] != 0) && return 8
    (x > localDomain[1] - dx) && (coords[1] != dims[1] - 1) && return 4
    return 0 # inside local domain, no neighbour
end
