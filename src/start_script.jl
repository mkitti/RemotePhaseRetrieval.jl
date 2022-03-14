println("Loading RemotePhaseRetrieval")

using CUDA
using RemotePhaseRetrieval

println("Warming up")
RemotePhaseRetrieval.warmup()

println("Starting server...")
RemotePhaseRetrieval.run_server()
println("Server started.")

function wait_for_quit()
    msg = ""
    while msg != "quit"
        println("Type \"quit\" to exit.")
        for d = 0:CUDA.ndevices()-1
            CUDA.device!(d)
            CUDA.reclaim()
        end
        GC.gc()
        msg = readline()
    end
end

wait_for_quit()