# AI-Guided Enzyme Variant Design & Stability Screening

An end-to-end pipeline that proposes stabilizing point mutations for a therapeutic enzyme using modern AI tools - a protein language model (ESM-2), a structure predictor (ESMFold), and a neural network potential (MACE) - instead of classical, expensive methods (wet-lab mutagenesis screens or full DFT/MD simulation).

## Why this matters

Enzyme stability is a precondition for using a biological catalyst outside a sterile lab: industrial biocatalysis, detergents, diagnostics, bioremediation - and, the angle taken here, **enzymes as drugs**. A drug that degrades or gets cleared from the body too quickly needs more frequent dosing, and can trigger immune reactions along the way. Improving intrinsic protein stability is one lever for addressing that, complementary to established approaches like PEGylation.

## Target enzyme: E. coli L-asparaginase II (PDB [3ECA](https://www.rcsb.org/structure/3ECA))

L-asparaginase II is a real oncology drug (marketed as *Elspar*), used to treat acute lymphoblastic leukemia (ALL). It catalyzes the hydrolysis of L-asparagine into L-aspartate + ammonia. Most healthy cells synthesize their own asparagine and don't need the serum supply; many leukemic cells in ALL have lost that ability and depend on scavenging it from the blood. Depleting serum asparagine selectively starves the cancer cells.

Its clinical limitation is genuine and well documented: short serum half-life, immunogenicity (it's a bacterial protein), and susceptibility to proteolytic degradation. That makes it a real test case for AI-guided stability engineering, not a toy example.

A few structural facts worth knowing before reading the results below:
- The protein is a **homotetramer** (chains A/B/C/D, all identical sequence) - a **"dimer of dimers"**, not a symmetric ring: chain A shares a large, tight interface with chain C (~65 contacting residues), and much smaller, roughly-equal secondary interfaces with B and D (~28-30 residues each).
- The active site is formed **within each chain**, not between two different catalytic residues on different chains: the two known catalytic residues, Thr12 and Thr89, sit ~7.5 Å apart *within the same chain* - one shared active site per chain, so **4 chains → 4 active sites**, not 8. But each site's pocket is only fully formed with help from the tight-dimer partner chain, which is why the whole inter-subunit interface (111 of 326 chain-A residues, ~34%) matters for mutation safety, not just the two catalytic residues themselves.

## Pipeline

Four Jupyter notebooks, each consuming the previous one's output:

| # | Notebook | What it does | GPU needed? |
|---|---|---|---|
| 01 | [`target_selection`](notebooks/01_target_selection.ipynb) | Downloads 3ECA from the PDB, confirms the tetramer, extracts the chain A sequence from resolved coordinates, sanity-checks composition and the catalytic residues, saves a reference FASTA. | No |
| 02 | [`esm_mutation_scoring`](notebooks/02_esm_mutation_scoring.ipynb) | Zero-shot masked-marginal scoring with ESM-2 (650M): masks each position in turn and scores every possible substitution by how "evolutionarily natural" the model finds it (Meier et al., 2021). Flags substitutions at the catalytic residues. | Yes |
| 03 | [`structure_prediction`](notebooks/03_structure_prediction.ipynb) | Folds the wild type and the top ESM-2 candidates with ESMFold (via the free ESM Atlas API), checks fold confidence (pLDDT), and computes the *real* inter-subunit interface from the notebook-01 crystal structure to flag any candidate sitting at a subunit contact. | No |
| 04 | [`nnp_stability_scoring`](notebooks/04_nnp_stability_scoring.ipynb) | Grafts each mutation onto a shared reference structure (Kabsch superposition), relaxes the local pocket around it with the MACE-OFF23 neural network potential, and compares atomization energy against the wild type as a ΔΔG-like stability proxy. Combines all three signals into a final ranking, with a py3Dmol visualization of the top candidates. | Recommended |

Each notebook is self-contained, runs in Google Colab, and reads/writes to `data/` and `results/` on Google Drive so outputs carry over automatically from one notebook to the next.

## Results

Final combined ranking (`results/variant_rankings.csv`), sorted by the physics-based stability signal:

| Mutation | NNP energy Δ (eV) | ESM-2 score | At subunit interface | Near terminus |
|---|---|---|---|---|
| **A266I** | **-19.2** (most favorable) | 3.88 | no | no |
| **A266V** | **-15.9** | 4.45 | no | no |
| A228V | -4.3 | 3.49 | no | no |
| C105S | -2.6 | 6.37 | no | no |
| L1M | +1.6 (~neutral) | 9.32 (highest, but likely an artifact) | no | **yes** |
| Y250N | +26.5 | 5.66 | **yes** | no |
| Y250P | +28.2 | 3.87 | **yes** | no |
| Y250S | +32.9 (least favorable) | 4.89 | **yes** | no |

**Top pick: A266I** (Ala266Ile), with **A266V** (Ala266Val) a close second. These are the only two candidates where all three independent signals - ESM-2's evolutionary-tolerance score, ESMFold's structural confidence, and MACE's local physics-based energy - agree favorably at once.

Two findings worth calling out:
- **`L1M`'s top ESM-2 score looks like an artifact, not a real signal.** It sits at the very N-terminus (flagged in notebook 03 as a region prone to inflated scores, since termini are less structurally constrained), and its physics-based energy is essentially neutral - two independent checks now agree it shouldn't be trusted despite topping the ESM-2-only ranking.
- **`Y250N/S/P` are the weakest candidates**, and for two independent reasons that agree: they sit right at the real inter-subunit interface (a geometric fact from the crystal structure, not just a catalytic-residue proximity check), *and* they have the worst physics-based energies in the whole set.

## Getting started

### Run everything in Google Colab (recommended)

All four notebooks install their own dependencies inline (`%pip install ...`) and mount Google Drive, so no local setup is needed.

1. Upload this project folder to Google Drive (e.g. `MyDrive/enzyme-design-ai/`), keeping the `data/`, `notebooks/`, and `results/` structure.
2. Open a notebook from Google Drive directly - right-click the `.ipynb` file → **Open with Google Colaboratory** (more reliable than Colab's own "Open notebook" file picker for freshly-uploaded files).
3. The first code cell mounts your Drive and `cd`s into the project folder - edit the `PROJECT_DIR` variable there if your folder name/location differs.
4. For notebooks 02 and 04: `Runtime` → `Change runtime type` → select a **T4 GPU** (notebook 04 also works on CPU, just slower). Notebooks 01 and 03 don't need a GPU.
5. `Runtime` → `Run all`.
6. Run the notebooks in order (01 → 02 → 03 → 04) - since they all read/write through the same mounted Drive folder, each notebook's outputs are immediately available to the next with no manual download/upload step.

### Run notebooks 01 and 03 locally instead (optional)

These two are lightweight (Biopython/pandas only) and don't need a GPU, so they can also run outside Colab if you prefer:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip on macOS/Linux
```

In VS Code: open the notebook, pick the `.venv` interpreter as the kernel, run cells top to bottom. Notebooks 02 and 04 still need Colab (or a local GPU + PyTorch/mace-torch install, not covered here).

## Repo structure

```
.
├── README.md
├── requirements.txt          # local, lightweight deps for notebooks 01/03 (biopython, pandas, ipykernel)
├── data/
│   ├── 3eca_chainA.fasta     # reference sequence (notebook 01 output)
│   └── pdb/pdb3eca.ent       # downloaded crystal structure (notebook 01 output)
├── notebooks/
│   ├── 01_target_selection.ipynb
│   ├── 02_esm_mutation_scoring.ipynb
│   ├── 03_structure_prediction.ipynb
│   └── 04_nnp_stability_scoring.ipynb
└── results/
    ├── esm2_mutation_scores.csv   # notebook 02 output - full scored mutation table
    ├── structure_scores.csv      # notebook 03 output - + pLDDT, interface flags
    ├── structures/                # notebook 03 output - ESMFold-predicted PDBs
    ├── structures_grafted/        # notebook 04 output - mutations grafted onto WT scaffold
    └── variant_rankings.csv       # notebook 04 output - final combined ranking
```

## Limitations

- **ESMFold predicts each chain as an isolated monomer.** A good pLDDT score confirms the mutated sequence folds into a well-defined structure on its own - it does not confirm the tetramer still assembles correctly. The interface flag (from the real crystal geometry) is how this pipeline compensates for that blind spot, but it's a proxy, not a direct check.
- **MACE-OFF23 is a general organic-chemistry potential, not protein-specific.** It's a fast, practical off-the-shelf choice for this kind of exploratory local relaxation, not a substitute for a protein-specific force field or full QM treatment.
- **The NNP relaxation is local (pocket-only) and monomer-only.** It does not evaluate whether an interface-adjacent mutation (like the rejected `Y250N/S/P`) still permits correct tetramer assembly - the same Kabsch-grafting technique used here could in principle be extended to the tetramer coordinates for that check, but that's not implemented in this pipeline.
- **All three signals are proxies, not ground truth.** ESM-2's score reflects evolutionary tolerance across homologous sequences, not stability directly. ESMFold's pLDDT is the model's own confidence, not an experimental measurement. The NNP energy is a fast, local approximation to a true ΔΔG calculation. Agreement across independent proxies (as seen for A266I/A266V) is a reasonable basis for prioritizing candidates for further study - not a substitute for experimental validation.

## References

- Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science, 379(6637), 1123–1130. DOI: [10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574) - ESM-2 and ESMFold.
- Meier, J. et al. (2021). *Language models enable zero-shot prediction of the effects of mutations on protein function.* NeurIPS 2021, vol. 34, pp. 29287–29303 - the masked-marginal zero-shot scoring method used in notebook 02.
- Batatia, I. et al. (2022). *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields.* NeurIPS 2022, vol. 35, pp. 11423–11436.
- Swain, A.L. et al. (1993). *Crystal structure of Escherichia coli L-asparaginase, an enzyme used in cancer therapy.* PNAS - structural basis for PDB 3ECA.
- Dauparas, J. et al. (2022). *Robust deep learning-based protein sequence design using ProteinMPNN.* Science, 378(6615), 49–56. DOI: [10.1126/science.add2187](https://doi.org/10.1126/science.add2187) - not used in this pipeline, listed as a possible cross-validation method for future work.
