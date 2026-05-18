# $\text{Riemann }1D$

Exact and approximate Riemann solvers for 1D shock tube problems, written in Julia.

> **Note:** This is a personal demo project created solely for learning, experimentation, and hands-on practice. It is _NOT_ intended for production use. Long-term maintenance, code quality, stability, and security updates are not guaranteed.

To build dependencies, run following command from repo root:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

To run `run_sod.jl`：

```bash
julia --project=examples -e 'import Pkg; Pkg.develop(path=".")' # initial run
```

After building `Manifest.toml`, simply run:

```bash
julia --project=examples examples/run_sod.jl
```

To build documents:

```bash
julia --project=docs -e 'import Pkg; Pkg.develop(path=".")' # initial run
julia --project=docs docs/make.jl
```