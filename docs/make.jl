using CFTransport
using Documenter

DocMeta.setdocmeta!(CFTransport, :DocTestSetup, :(using CFTransport); recursive=true)

makedocs(;
    modules=[CFTransport],
    authors="The ClimFlows contributors",
    sitename="CFTransport.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/CFTransport.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/CFTransport.jl",
    devbranch="main",
)
