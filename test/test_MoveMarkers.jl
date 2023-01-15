using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = ENV["USE_GPU"] == "true" ? true : false

@static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 2)
else
        @init_parallel_stencil(Threads, Float64, 2)
end

include("../scripts/MoveMarkers.jl")
include("../scripts/GlobalGather.jl")

using Test

# TEST 1: MOVE IN CONSTANT VELOCITY FIELD --------------------------------------------------------------
Nx = 10
Ny = 13
dx = 0.4
dy = 0.7
lx = (Nx - 1) * dx;
ly = (Ny - 1) * dy;
# marker coords
Nm = 25 * (Nx - 1) * (Ny - 1)
x_m = @rand(Nm) .* (lx - dx) .+ 0.5dx
y_m = @rand(Nm) .* (ly - dy) .+ 0.5dy
x_m_initial = Array(x_m)
y_m_initial = Array(y_m)

Vx = @ones(Nx + 2, Ny + 1)
Vy = @ones(Nx + 1, Ny + 2) .* (-2)

dt = 0.5 * min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)))

x_vx_min = -dx
y_vx_min = -0.5dy
x_vy_min = -0.5dx
y_vy_min = -dy

moveMarkersRK4!(x_m, y_m, Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dt, lx, ly, dx, dy)

# manually move markers in constant V field
x_m_manual = x_m_initial .+ 1.0 .* dt # Vx = 1
y_m_manual = y_m_initial .+ -2.0 .* dt # Vy = -2

@test all(x_m_manual .≈ Array(x_m))
@test all(y_m_manual .≈ Array(y_m))


# TEST 2: compare velocity interpolation with manual computation in a single cell ----------------------
#         (In the actual GPU code, the functions tested here are device functions.
#          However, here they will be just compiled for the CPU, since they are not
#          called from a Kernel.)
Nx = 2
Ny = 2
Vx = rand(Nx + 2, Ny + 1)
Vy = rand(Nx + 1, Ny + 2)
# define four points, one each in bottom right, bottom left, top right & top left part of the cell
x = (0.99rand(4) .+ 0.005 .+ [0, 0, 1, 1]) .* 0.5dx
y = (0.99rand(4) .+ 0.005 .+ [0, 1, 0, 1]) .* 0.5dy

# 1) BOTTOM LEFT
# Vx
ix, iy, dxij, dyij = topleftIndexRelDist(x_vx_min, y_vx_min, x[1], y[1], dx, dy)
#@test ix == 2 && iy == 1
#@test dxij ≈ x[1]/dx && dyij ≈ (0.5dy+y[1])/dy
vxref = (1 - dxij) * (1 - dyij) * Vx[2, 1] +
        dxij * (1 - dyij) * Vx[3, 1] +
        dxij * dyij * Vx[3, 2] +
        (1 - dxij) * dyij * Vx[2, 2] +
        0.5 * (dxij - 0.5)^2 * ((1 - dyij) * (Vx[1, 1] - 2Vx[2, 1] + Vx[3, 1]) + dyij * (Vx[1, 2] - 2Vx[2, 2] + Vx[3, 2]))
# Vy
ix, iy, dxij, dyij = topleftIndexRelDist(x_vy_min, y_vy_min, x[1], y[1], dx, dy)
#@test ix == 1 && iy == 2
#@test dxij ≈ (0.5dx+x[1])/dx && dyij ≈ y[1]/dy
vyref = (1 - dxij) * (1 - dyij) * Vy[1, 2] +
        dxij * (1 - dyij) * Vy[2, 2] +
        dxij * dyij * Vy[2, 3] +
        (1 - dxij) * dyij * Vy[1, 3] +
        0.5 * (dyij - 0.5)^2 * ((1 - dxij) * (Vy[1, 1] - 2Vy[1, 2] + Vy[1, 3]) + dxij * (Vy[2, 1] - 2Vy[2, 2] + Vy[2, 3]))
# check interpolated velocities
vx, vy = interpolateV(x[1], y[1], Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)
@test vx ≈ vxref && vy ≈ vyref

# 2) TOP LEFT
# Vx
ix, iy, dxij, dyij = topleftIndexRelDist(x_vx_min, y_vx_min, x[2], y[2], dx, dy)
@test ix == 2 && iy == 2
@test dxij ≈ x[2] / dx && dyij ≈ (-0.5dy + y[2]) / dy
vxref = (1 - dxij) * (1 - dyij) * Vx[2, 2] +
        dxij * (1 - dyij) * Vx[3, 2] +
        dxij * dyij * Vx[3, 3] +
        (1 - dxij) * dyij * Vx[2, 3] +
        0.5 * (dxij - 0.5)^2 * ((1 - dyij) * (Vx[1, 2] - 2Vx[2, 2] + Vx[3, 2]) + dyij * (Vx[1, 3] - 2Vx[2, 3] + Vx[3, 3]))
# Vy
ix, iy, dxij, dyij = topleftIndexRelDist(x_vy_min, y_vy_min, x[2], y[2], dx, dy)
@test ix == 1 && iy == 2
@test dxij ≈ (0.5dx + x[2]) / dx && dyij ≈ y[2] / dy
vyref = (1 - dxij) * (1 - dyij) * Vy[1, 2] +
        dxij * (1 - dyij) * Vy[2, 2] +
        dxij * dyij * Vy[2, 3] +
        (1 - dxij) * dyij * Vy[1, 3] +
        0.5 * (dyij - 0.5)^2 * ((1 - dxij) * (Vy[1, 2] - 2Vy[1, 3] + Vy[1, 4]) + dxij * (Vy[2, 2] - 2Vy[2, 3] + Vy[2, 4]))
# check interpolated velocities
vx, vy = interpolateV(x[2], y[2], Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)
@test vx ≈ vxref && vy ≈ vyref

# 3) BOTTOM RIGHT
# Vx
ix, iy, dxij, dyij = topleftIndexRelDist(x_vx_min, y_vx_min, x[3], y[3], dx, dy)
@test ix == 2 && iy == 1
@test dxij ≈ x[3] / dx && dyij ≈ (0.5dy + y[3]) / dy
vxref = (1 - dxij) * (1 - dyij) * Vx[2, 1] +
        dxij * (1 - dyij) * Vx[3, 1] +
        dxij * dyij * Vx[3, 2] +
        (1 - dxij) * dyij * Vx[2, 2] +
        0.5 * (dxij - 0.5)^2 * ((1 - dyij) * (Vx[2, 1] - 2Vx[3, 1] + Vx[4, 1]) + dyij * (Vx[2, 2] - 2Vx[3, 2] + Vx[4, 2]))
# Vy
ix, iy, dxij, dyij = topleftIndexRelDist(x_vy_min, y_vy_min, x[3], y[3], dx, dy)
@test ix == 2 && iy == 2
@test dxij ≈ (-0.5dx + x[3]) / dx && dyij ≈ y[3] / dy
vyref = (1 - dxij) * (1 - dyij) * Vy[2, 2] +
        dxij * (1 - dyij) * Vy[3, 2] +
        dxij * dyij * Vy[3, 3] +
        (1 - dxij) * dyij * Vy[2, 3] +
        0.5 * (dyij - 0.5)^2 * ((1 - dxij) * (Vy[2, 1] - 2Vy[2, 2] + Vy[2, 3]) + dxij * (Vy[3, 1] - 2Vy[3, 2] + Vy[3, 3]))
# check interpolated velocities
vx, vy = interpolateV(x[3], y[3], Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)
@test vx ≈ vxref && vy ≈ vyref

# 4) TOP RIGHT
# Vx
ix, iy, dxij, dyij = topleftIndexRelDist(x_vx_min, y_vx_min, x[4], y[4], dx, dy)
@test ix == 2 && iy == 2
@test dxij ≈ x[4] / dx && dyij ≈ (-0.5dy + y[4]) / dy
vxref = (1 - dxij) * (1 - dyij) * Vx[2, 2] +
        dxij * (1 - dyij) * Vx[3, 2] +
        dxij * dyij * Vx[3, 3] +
        (1 - dxij) * dyij * Vx[2, 3] +
        0.5 * (dxij - 0.5)^2 * ((1 - dyij) * (Vx[2, 2] - 2Vx[3, 2] + Vx[4, 2]) + dyij * (Vx[2, 3] - 2Vx[3, 3] + Vx[4, 3]))
# Vy
ix, iy, dxij, dyij = topleftIndexRelDist(x_vy_min, y_vy_min, x[4], y[4], dx, dy)
@test ix == 2 && iy == 2
@test dxij ≈ (-0.5dx + x[4]) / dx && dyij ≈ y[4] / dy
vyref = (1 - dxij) * (1 - dyij) * Vy[2, 2] +
        dxij * (1 - dyij) * Vy[3, 2] +
        dxij * dyij * Vy[3, 3] +
        (1 - dxij) * dyij * Vy[2, 3] +
        0.5 * (dyij - 0.5)^2 * ((1 - dxij) * (Vy[2, 2] - 2Vy[2, 3] + Vy[2, 4]) + dxij * (Vy[3, 2] - 2Vy[3, 3] + Vy[3, 4]))
# check interpolated velocities
vx, vy = interpolateV(x[4], y[4], Vx, Vy, x_vx_min, y_vx_min, x_vy_min, y_vy_min, dx, dy)
@test vx ≈ vxref && vy ≈ vyref
