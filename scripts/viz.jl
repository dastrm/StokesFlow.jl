# run: julia viz.jl
using Plots, Plots.Measures, Printf, MAT

const dims = [4, 2] # number of MPI processes
const nt = 1 # number of time steps

function viz_markers()
  nprocs = dims[1] * dims[2]
  anim = @animate for it = 1:nt
    x_m = []
    y_m = []
    ρ_m = []
    for ip = 1:nprocs
      file = matopen(string(@sprintf("viz_out/markers_%04d_%04d", it - 1, ip - 1), ".mat"))
      x_loc = read(file, "x_m")
      y_loc = read(file, "y_m")
      ρ_loc = read(file, "rho_m")
      close(file)
      x_m = append!(x_m, x_loc)
      y_m = append!(y_m, y_loc)
      ρ_m = append!(ρ_m, ρ_loc)
    end
    @assert size(x_m, 1) == size(y_m, 1)
    @assert size(x_m, 1) == size(ρ_m, 1)

    opts = (color=Int.(round.(exp.(ρ_m))), aspect_ratio=1, size=(1200, 1150), framestyle=:box, label=false, grid=false, margin=1mm, yflip=true, legend=false, markersize=3, markerstrokewidth=0, cbar=true)
    scatter(x_m, y_m; opts...)
  end

  gif(anim, "viz_out/markers.gif"; fps=1)

  return nothing
end

viz_markers()
