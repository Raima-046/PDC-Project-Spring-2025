# PDC-Project-Spring-2025

# Parallel SSSP Project

This project implements and analyzes the **Single Source Shortest Path (SSSP)** algorithm using different parallel approaches.

## Files

* **Serial\_sssp.cpp** – Standard serial implementation of SSSP.
* **serialOMP.cpp** – SSSP using OpenMP for parallelization.
* **serial\_metis.cpp** – SSSP leveraging METIS graph partitioning.
* **serial\_metisMPI.cpp** – METIS + MPI-based parallel SSSP.
* **openMPMPI.cpp** – Combined OpenMP + MPI implementation.

(All files are in C++.)

## Datasets

* **AI-generated Dataset**: Provided in `.txt` format.
* **Internet Dataset**: In `.mtx` format (Matrix Market format).

## Python Scripts

* **txt\_to\_graph.py** – Converts the `.txt` dataset into a graph structure.
* **mtx\_to\_graph.py** – Converts the `.mtx` dataset into a graph.

## Execution

* The MPI-based programs were tested on a **cluster setup** as well as on a **local PC cluster environment**.

## Report

* The **report.pdf** contains:

  * Time analysis of all implementations.
  * Graphs and tables showing CPU utilization and performance.
  * A brief project description and setup notes.

## Dependencies & Setup

We used **METIS** and **GKlib** for graph partitioning. To set up:

```bash
git clone https://github.com/KarypisLab/METIS.git
git clone https://github.com/KarypisLab/GKlib.git
```
