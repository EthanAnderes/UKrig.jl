using Documenter, UKrig

makedocs(;
    modules=[UKrig],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/EthanAnderes/UKrig.jl/blob/{commit}{path}#L{line}",
    sitename="UKrig.jl",
    authors="Ethan Anderes",
    assets=String[],
)

deploydocs(;
    repo="github.com/EthanAnderes/UKrig.jl",
)
