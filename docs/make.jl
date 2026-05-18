using Riemann1D
using Documenter

DocMeta.setdocmeta!(Riemann1D, :DocTestSetup, :(using Riemann1D); recursive=true)

makedocs(
    modules = [Riemann1D],
    authors = "NCC79601 <ncc79601@163.com>",
    sitename = "Riemann1D.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        collapselevel = 1,
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/NCC79601/Riemann1D.git",
    devbranch = "dev",
)
