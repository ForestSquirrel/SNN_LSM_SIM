# SNN_LSM_SIM
# ⚠️🔥 **THIS SHIT IS WIP AF — ALWAYS BETA ALWAYS ~~BETTER~~ ** 🔥⚠️

This project is a **Spiking Neural Network (SNN)** and **Liquid State Machine (LSM)** simulator built in C++/CUDA. It’s meant for research / experimentation on spiking neural architectures and GPU‑accelerated simulation.

## 🚀 Features

- CUDA accelerated simulation core 
- Same syntax for SNN/LSM

## 📁 Repo Structure

```
/.github/workflows/   CI configs
/core/                core simulation source code
/doxide.yaml          documentation generator config
/mkdocs.yaml          docs site config
/docs/                generated docs content (Markdown)
/site/                generated static site files for docs hosting
```

##  Getting Started

### Requirements  
### Requirements  
- CUDA >= 12.2  
- C++ >= 14 compiler compatible with CUDA  
- Thrust (bundled with CUDA)  
- cuBLAS & cuSPARSE (CUDA math libraries bundled with CUDA)  
- [Eigen3 Library](https://libeigen.gitlab.io/) for 3D LSM layout generation


### Build and Run  
```bash
# Assuming you have a .sln or project set up:
#   on Windows (Visual Studio) open `SNN_CUDA_SIM.sln` and build
```


## 📚 Documentation

The full documentation is published via GitHub Pages.  
[**📖 Visit the docs site**](https://ForestSquirrel.github.io/SNN_LSM_SIM/)  
