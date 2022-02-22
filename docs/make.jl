using RemotePhaseRetrieval
using Documenter

DocMeta.setdocmeta!(RemotePhaseRetrieval, :DocTestSetup, :(using RemotePhaseRetrieval); recursive=true)

makedocs(;
    modules=[RemotePhaseRetrieval],
    authors="Mark Kittisopikul <kittisopikulm@janelia.hhmi.org> and contributors",
    repo="https://github.com/kittisopikulm@janelia.hhmi.org/RemotePhaseRetrieval.jl/blob/{commit}{path}#{line}",
    sitename="RemotePhaseRetrieval.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kittisopikulm@janelia.hhmi.org.github.io/RemotePhaseRetrieval.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kittisopikulm@janelia.hhmi.org/RemotePhaseRetrieval.jl",
    devbranch="main",
)
