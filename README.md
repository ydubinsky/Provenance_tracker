# Provenance Tracker for Data Cleaning Pipelines (Pandas)
This project implements a lightweight provenance tracker that wraps common Pandas
cleaning operations, logs pre/post metadata and diffs, and exports both JSON and RDF.

## Features
- Transparent wrappers for `dropna`, `fillna`, `rename`, `merge`, and a generic `apply()`.
- Operation-level logging: parameters, shapes, column changes, hashes, missingness stats.
- Diff summaries (row deltas, changed cells).
- JSON export (`tracker.save_json()`), RDF export to PROV-O (`tracker.save_rdf()` for generating the rdf file).
- Simple DOT graph export of the pipeline (`tracker.save_graph()`).
- Missing markers configurable (treats `'?'`, `' ?'`, `'NA'`, `'N/A'`, `'nan'`, `'NULL'` as missing by default).

## Install
```bash
pip install pandas rdflib
```

## Computational Environment
This project was developed and tested using Python 3.13.2. libraries used are:
- `pandas`
- `numpy`
- `graphviz` (requires system-level installation of Graphviz; e.g., `brew install graphviz` on macOS, `sudo apt-get install graphviz` on Debian/Ubuntu)
- `rdflib`
- `ucimlrepo`

The recommended operating systems for running this project are Linux or macOS.

## How-To-Run Instructions

To run the example in this project, follow these steps:

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ensure Graphviz is installed on your system.**
    *   **macOS (using Homebrew):** `brew install graphviz`
    *   **Debian/Ubuntu:** `sudo apt-get install graphviz`
    *   For Windows, please refer to the official Graphviz website for installer.
4.  **Open the main example notebook:**
    Open `provenance_demo.ipynb` in a Jupyter-compatible environment (e.g., JupyterLab, VS Code with Python extension).
5.  **Run all cells:**
    Execute all cells in the notebook sequentially.
6.  **Expected Outputs:**
    The notebook should generate provenance logs (JSON and RDF), and a DOT graph of the pipeline.
7.  **Writing Errors**
    In case of writing errors, please delete the original output files provenance_census_graph.dot, provenance_census_log.json, provenance.ttl (This is done to prevent tampering of generated files)
