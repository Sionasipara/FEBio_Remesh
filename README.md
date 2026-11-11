# FEBio Breast Model Remeshing

A Python-based pipeline for generating uniform surface and volumetric meshes of patient-specific 3D breast models for finite element analysis (FEA) using **Gmsh and FEBio**.

The script improves mesh uniformity while preserving anatomical tissue classifications, reconstructing boundary conditions, and retaining simulation configurations from the original FEBio model.

## Overview

Finite element simulations require high-quality volumetric meshes to accurately model tissue deformation and displacement.

This project automates the remeshing of an existing FEBio breast model, generating near-equilateral surface triangles and optimized tetrahedral elements while maintaining the model's anatomical and simulation structure.

The pipeline supports three tissue regions:

- Adipose tissue
- Fibroglandular tissue
- Tumour

## Remeshing Pipeline

**1. Load FEBio Model**

Read the original `.feb` file and extract mesh nodes, tetrahedral elements, tissue regions, and boundary node sets.

**2. Extract Exterior Surface**

Identify the exterior triangular faces of the volumetric tetrahedral mesh and export the closed surface as an STL file.

**3. Generate Uniform Mesh Using Gmsh**

Generate a new surface and volumetric mesh with a configurable target edge length (default: 2 mm).

Gmsh is used to generate and optimize the tetrahedral elements.

**4. Transfer Tissue Classifications**

Assign each new tetrahedral element to its corresponding anatomical tissue region using source-cell containment, with nearest-centroid matching as a fallback.

**5. Reconstruct Boundary Conditions**

Geometrically reconstruct the ChestWall node set and two posterior anchor nodes. Rebuild the conforming triangular skin shell, excluding the chest wall.

**6. Export Updated FEBio Model**

Write the remeshed geometry into a new `.feb` file while retaining the original material definitions, solver settings, loads, boundary-condition definitions, and output requests.

**7. Generate Mesh Quality Report**

Calculate surface edge-length distributions, triangle quality, tetrahedral quality, and element counts for mesh evaluation.

## Requirements

- Python 3
- NumPy
- SciPy
- PyVista
- Gmsh Python API

Install the Python dependencies:

```bash
pip install numpy scipy pyvista gmsh
```

## Usage

Run the remeshing script from the command line:

```bash
python uniform_remesh_febio.py input.feb output.feb --target 2.0
```

Arguments:

| Argument | Description |
|---|---|
| `input` | Original FEBio model file |
| `output` | Destination remeshed FEBio model |
| `--target` | Target mesh edge length in millimetres (default: 2.0) |
| `--vendor` | Optional directory containing the Gmsh Python package |

## Outputs

The pipeline generates:

- **Remeshed `.feb` file:** Updated FEBio model containing the new volumetric mesh, tissue regions, and reconstructed boundary sets.
- **Surface `.stl` file:** Extracted exterior surface stored in `.remesh-cache/`.
- **Mesh quality `.txt` report:** Element counts, edge-length statistics, and mesh-quality metrics.

## Limitations

- The input must contain the expected FEBio mesh structure and named tissue regions.
- Tissue classifications are transferred using geometric matching and should be validated near tissue interfaces.
- Boundary nodes are reconstructed geometrically and should be checked before simulation.
- Existing simulation results cannot be reused because remeshing changes node and element identities.
- The remeshed model should be validated in FEBio before interpreting simulation results.

## Applications

This pipeline supports research involving:

- Patient-specific breast biomechanics
- Breast tissue deformation simulation
- Tumour displacement analysis
- Breast-conserving surgery modelling
- Finite element mesh preprocessing and optimization

## Technologies

**Python | Gmsh | FEBio | NumPy | SciPy | PyVista**
