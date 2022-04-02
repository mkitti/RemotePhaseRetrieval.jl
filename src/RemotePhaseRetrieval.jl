module RemotePhaseRetrieval

# Write your package code here.

using StaticArrays
using FFTW
using LsqFit
using CUDA
using HDF5
using Sockets

CUDA.allowscalar(false)

abstract type AbstractPSFWorkspace end

struct CudaPSFWorkspace{F} <: AbstractPSFWorkspace
    # Independent variables
    nx::Int
    ny::Int
    nz::Int
    napo::Int
    nxy::Int
    nzc::Int
    aperature::CuArray{F, 3} # Aperature, possibly with obscuration. Usually 41x41x6 (X x Y x Apodisations)
    aber::CuArray{Complex{F}, 3} # Abberations via function_aber (MATLAB). Usually 41x41x27 (X x Y x DOF), DOF = Z
    zComp::CuArray{F, 3} # Zernicke components. Usually 41x41x17 (X x Y x zC). zC = # of zernicke components, 1:16, 25
    zComp_flat::CuArray{F, 2} # zComp as (XY x zC)
    # Workspace
    phase::CuArray{F, 2} # From zernicke components and coefficients. Usually 41x41 (X x Y)
    phase_flat::CuArray{F, 2} # phase as (XY x 1)
    x::CuArray{Complex{F}, 3} # Aperature times e^(i*θ), θ = phase. Usually 41x41x6 (X x Y x Apodizations)
    x4::CuArray{Complex{F}, 4} # x with the Apodizations shifted into the 4th dimension
    x_aber::CuArray{Complex{F}, 4} # Aber times x
    intensity::CuArray{F, 4} # Intensity, unshifted after fft2
    summed_intensity::CuArray{F, 4} # Summed intensity, summed across 6 apodisations
    psf_vec::CuArray{F, 1} # Dedicated space on the GPU for storing this
    device::CuDevice
end

function CudaPSFWorkspace{F}(aperature, aber, zComp) where F
    nx   = size(aperature, 1) # 41
    ny   = size(aperature, 2) # 41
    nz   = size(aber, 3) # 27
    napo = size(aperature, 3) # 6
    nxy  = nx * ny # 41*41 = 1681
    nzc  = size(zComp, 3) # 17
    zComp_flat       =             reshape(zComp, nxy, nzc)
    phase            =         CuMatrix{F}(undef, nx,  ny)
    phase_flat       =             reshape(phase, nxy, 1)
    x                = CuArray{Complex{F}}(undef, nx,  ny, napo)
    x4               =             reshape(x, nx, ny,  1, napo)
    x_aber           = CuArray{Complex{F}}(undef, nx,  ny, nz, napo)
    intensity        =          CuArray{F}(undef, nx,  ny, nz, napo)
    summed_intensity =          CuArray{F}(undef, nx,  ny, nz, 1)
    psf_vec          =          CuArray{F}(undef, nx*ny*nz)
    # Fill to force eager allocation
    fill!(phase, 0.0)
    fill!(x, 0.0)
    fill!(x_aber, 0.0)
    fill!(intensity, 0.0)
    fill!(summed_intensity, 0.0)
    fill!(psf_vec, 0.0)
    CudaPSFWorkspace{F}(
        nx, ny, nz,
        napo, nxy, nzc,
        aperature,
        aber,
        zComp,
        zComp_flat,
        phase,
        phase_flat,
        x,
        x4,
        x_aber,
        intensity,
        summed_intensity,
        psf_vec,
        CUDA.device()
    )
end
CudaPSFWorkspace(aperature, aber, zComp) = CudaPSFWorkspace{Float64}(aperature, aber, zComp)

struct PSFWorkspace <: AbstractPSFWorkspace
    # Independent variables
    nx::Int
    ny::Int
    nz::Int
    napo::Int
    nxy::Int
    nzc::Int
    aperature::Array{Float64, 3} # Aperature, possibly with obscuration. Usually 41x41x6 (X x Y x Apodisations)
    aber::Array{ComplexF64, 3} # Abberations via function_aber (MATLAB). Usually 41x41x27 (X x Y x DOF), DOF = Z
    zComp::Array{Float64, 3} # Zernicke components. Usually 41x41x17 (X x Y x zC). zC = # of zernicke components, 1:16, 25
    zComp_flat::Matrix{Float64} # zComp as (XY x zC)
    # Workspace
    phase::Matrix{Float64} # From zernicke components and coefficients. Usually 41x41 (X x Y)
    phase_flat::Matrix{Float64} # phase as (XY x 1)
    x::Array{ComplexF64, 3} # Aperature times e^(i*θ), θ = phase. Usually 41x41x6 (X x Y x Apodizations)
    x4::Array{ComplexF64, 4} # x with the Apodizations shifted into the 4th dimension
    x_aber::Array{ComplexF64, 4} # Aber times x
    intensity::Array{Float64, 4} # Intensity, unshifted after fft2
    summed_intensity::Array{Float64, 4} # Summed intensity, summed across 6 apodisations
end

function PSFWorkspace(aperature, aber, zComp)
    nx   = size(aperature, 1) # 41
    ny   = size(aperature, 2) # 41
    nz   = size(aber, 3) # 27
    napo = size(aperature, 3) # 6
    nxy  = nx * ny # 41*41 = 1681
    nzc  = size(zComp, 3) # 17
    zComp_flat       =           reshape(zComp, nxy, nzc)
    phase            =   Matrix{Float64}(undef, nx,  ny)
    phase_flat       =           reshape(phase, nxy, 1)
    x                = Array{ComplexF64}(undef, nx,  ny, napo)
    x4               =               reshape(x, nx,  ny, 1, napo)
    x_aber           = Array{ComplexF64}(undef, nx,  ny, nz, napo)
    intensity        =    Array{Float64}(undef, nx,  ny, nz, napo)
    summed_intensity =    Array{Float64}(undef, nx,  ny, nz, 1)
    PSFWorkspace(
        nx, ny, nz,
        napo, nxy, nzc,
        aperature,
        aber,
        zComp,
        zComp_flat,
        phase,
        phase_flat,
        x,
        x4,
        x_aber,
        intensity,
        summed_intensity
    )
end

@inline function prepare_zernicke_coef(param, ::PSFWorkspace)
    SVector{17}(
        ntuple(length(param)) do i
            i == 1 ? 0 : param[i]
        end
    )
end

@inline function prepare_zernicke_coef(param, ::CudaPSFWorkspace)
    param = copy(param)
    param[1] = 0
    cua = CuArray(param)
    return cua
end

@inline fft2!(A) = fft!(A, (1,2))

function make_psfGenerator_vector_mk(w)
    function psfGenerator_vector_mk(psf_vec_g, _, param)
        I = param[1] / w.nxy
        zer_coef = prepare_zernicke_coef(param, w)
        w.phase_flat .= w.zComp_flat * zer_coef
        w.x .= w.aperature .* exp.(1im * w.phase)
        w.x_aber .= w.aber .* w.x4
        w.intensity .= abs2.(fft2!(w.x_aber))
        sum!(w.summed_intensity, w.intensity)
        psf_vec_g .= vec(w.summed_intensity) .* I
        return psf_vec_g
    end
end

function make_psfGenerator_vector_mk(w::CudaPSFWorkspace{F}) where F
    function psfGenerator_vector_mk(psf_vec_g, _, param)
        I = param[1] / w.nxy
        zer_coef = prepare_zernicke_coef(param, w)
        w.phase_flat .= w.zComp_flat * zer_coef
        w.x .= w.aperature .* exp.(1im * w.phase)
        w.x_aber .= w.aber .* w.x4
        w.intensity .= abs2.(fft2!(w.x_aber))
        sum!(w.summed_intensity, w.intensity)
        w.psf_vec .= vec(w.summed_intensity) .* I
        copyto!(psf_vec_g, w.psf_vec)
        return psf_vec_g
    end
end


#=
function psfGenerator_vector_mk(psf_vec_g, xdata, param)
    a = xdata_c.a
    aber=xdata_c.aber
    zComp=xdata_c.zComp
    I = param[1]
    zer_coef = prepare_zernicke_coef(param)
    phase_g .= reshape( reshape(zComp, size(zComp,1)*size(zComp,2), size(zComp,3)) * zer_coef,  size(zComp, 1), size(zComp, 2) )
    x_g .= a .* exp.(1im * phase_g)
    x_sz = (size(x_g,1), size(x_g,2), 1, size(x_g,3))
    cc .= aber .* reshape(x_g, x_sz)
    I_g .= abs2.(fft2!(cc))
    sum!(S_g, I_g)
    psf_vec_g .= vec(S_g) .* (I ./ (x_sz[1] * x_sz[2]))
    return psf_vec_g
end
=#

function run_server(addr = IPv4(0), port = 2009)
    server_task = @async begin
        server = listen(addr, port)
        for i in 1:Threads.nthreads()
            Threads.@spawn begin
                taskid = i
                task_local_storage("taskid", taskid)
                while true
                    println("Accepting connections on task $taskid")
                    sock = nothing
                    try
                        sock = accept(server)
                        process_connection(sock, taskid)
                    catch err
                        Base.printstyled("ERROR: "; color=:red, bold=true)
                        Base.showerror(stdout, err, Base.catch_backtrace())
                    finally
                        if !isnothing(sock) && isopen(sock)
                            close(sock)
                        end
                    end
                end
            end
        end
    end
    errormonitor(server_task)
    return server_task
end

function process_connection(sock, taskid)
    dev = cw[taskid].device
    device!(dev)
    while isopen(sock)
        try
            println("Waiting for command: task $taskid using $dev.")
            command = readline(sock)
            println("Received command $command: task $taskid using $dev.")
            if command == "lsq_fit_psf"
                write(sock, Int64(1))
                serve_lsq_fit_psf(sock, taskid)
            elseif command == "set_support_arrays"
                write(sock, Int64(1))
                serve_support_arrays(sock, taskid)
            elseif command == "reclaim"
                write(sock, Int64(1))
                CUDA.reclaim()
            elseif command == "close"
                if isopen(sock)
                    write(sock, Int64(1))
                end
                close(sock)
            else
                if isopen(sock)
                    write(sock, Int64(-1))
                end
                @warn "Unknown command received" command
            end
        catch err
            #println(err)
            if isopen(sock)
                write(sock, Int64(-1))
                close(sock)
            end
            rethrow(err)
        finally
        end
    end
end

function read_array(sock)
    println("Reading array")
    class = readline(sock)
    len = read(sock, Int64)
    ndims = read(sock, Int64)
    dims = ntuple(ndims) do i
        read(sock, Int64)
    end
    type = class == "double" || class == "Float64" ? Float64 : error("Unknown type")
    A = Array{type}(undef, dims)
    read!(sock, A)
    write(sock, sizeof(A))
    @info "Read array" class dims
    return A
end

function serve_lsq_fit_psf(sock, taskid)
        A = read_array(sock)
        result = processArray(A, taskid)
        write(sock, result.param...)
        println("$taskid: $(result.param)")
        CUDA.reclaim()
        GC.gc()
end

function serve_support_arrays(sock, taskid)
    println("Loading new support arrays.")
    a = read_array(sock)
    aber_real = read_array(sock)
    aber_imag = read_array(sock)
    aber = aber_real + 1im*aber_imag
    zComp = read_array(sock)
    #lock(locks[taskid])
    device!((taskid-1) % CUDA.ndevices())
    cw[taskid] = CudaPSFWorkspace(a, aber, zComp)
    #unlock(locks[taskid])
    println("Loaded new support arrays on task $taskid: $(size(aber)).")
end

const NUM_PARAMETERS = 17

ifftshift2(A) = ifftshift(ifftshift(A, 1), 2)

const cw = Vector{CudaPSFWorkspace}(undef, 1)
#const locks = Vector{ReentrantLock}(undef, 1)

function processArray(psf, taskid)
    param0 = map(i->i== 1 ? sum(psf)/length(psf) : 0.0, 1:NUM_PARAMETERS)
    lb = map(i->i==1 ? 0.0 : -10.0, 1:NUM_PARAMETERS)
    ub = map(i->i==1 ? 1.0 :  10.0, 1:NUM_PARAMETERS)
    x_tol = 1e-3
    #result = lock(locks[taskid]) do
    result = begin
        device!(cw[taskid].device)
        objfun = make_psfGenerator_vector_mk(cw[taskid])
        working_psf = ifftshift2(psf)[:]
        println("$taskid: aber, $(size(cw[taskid].aber)); psf, $(size(psf))")
        result = curve_fit(objfun, [], copy(working_psf), param0; inplace = true, x_tol, lower = lb, upper = ub, show_trace = false)
        fit_psf = objfun(copy(working_psf), nothing, result.param)
        residual = sum(abs2.(working_psf .- fit_psf))
        println("$taskid: residual = $residual")
        result
    end
    return result
end

function loadSupportArrays()
    h5open(joinpath(@__DIR__, "..", "mock_psf_test.h5")) do f
        (
            a = f["a"][],
            aber = f["aber"][],
            zComp = f["zComp"][]
        )
    end
end

function warmup()
    psf = h5open(joinpath(@__DIR__, "..", "mock_psf_test.h5")) do f
        f["psf"][]
    end
    out = nothing
    t = Threads.@spawn begin
        out = processArray(psf, 1)
        nothing
    end
    wait(t)
    return out
end

function send_array(t, array)
    # t = connect(2009)
    println(t, string(eltype(array)))
    write(t, length(array));
    write(t, ndims(array));
    write.(Ref(t), size(array));
    write(t, array)
    # bytes acknowledged
    bytes_ack = read(t, Int64)
    @assert bytes_ack == sizeof(array) "$bytes_ack $(sizeof(array))"
    return bytes_ack
end

function demo_client(t = connect(2009))
    if bytesavailable(t) > 0
        readavailable(t)
    end
    println(t, "lsq_fit_psf");
    status = read(t, Int64);
    @assert(status == 1);
    psf = h5open(joinpath(@__DIR__, "..", "mock_psf_test.h5")) do f
        Float64.(f["psf"][])
    end
    send_array(t, psf);
    buffer = zeros(Float64, 17)
    read!(t, buffer)
    println(t, "close")
    sleep(1)
    close(t)
    return buffer
end

function __init__()
    a, aber, zComp = loadSupportArrays()
    device!(0)
    cw[1] = CudaPSFWorkspace(a, aber, zComp)
    #locks[1] = ReentrantLock()
    Threads.resize_nthreads!(cw)
    #Threads.resize_nthreads!(locks)
    Threads.@threads for i=2:Threads.nthreads()
        d = (i-1) % CUDA.ndevices()
        device!(d)
        cw[i] = CudaPSFWorkspace(a, aber, zComp)
        println("$i: $d")
        println(cw[i].device)
    end
end

end # module RemotePhaseRetrieval
