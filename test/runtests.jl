using Test
import MPI

# This test driver is inspired by the tests from MPI.jl

testdir = @__DIR__
istest(f) = endswith(f, ".jl") && startswith(f, "test_")
testfiles = sort(filter(istest, readdir(testdir)))

nprocs = 4
USE_GPU = false

@testset "$f" for f in testfiles
    MPI.mpiexec() do mpirun
        cmd(n=nprocs) = `$mpirun -n $n $(Base.julia_cmd()) $(joinpath(testdir, f))`

        withenv("USE_GPU" => USE_GPU) do
            r = run(ignorestatus(cmd()))
            @test success(r)
        end
        
        
        #=
        if f == "test_spawn.jl"
            # Some command as the others, but always use a single process
            r = run(ignorestatus(cmd(1)))
            @test success(r)
        elseif f == "test_threads.jl"
            withenv("JULIA_NUM_THREADS" => "4") do
                r = run(ignorestatus(cmd()))
            end
            @test success(r)
        elseif f == "test_error.jl"
            r = run(ignorestatus(cmd()))
            @test !success(r)
        else
            r = run(ignorestatus(cmd()))
            @test success(r)
        end
        =#
    end
end
