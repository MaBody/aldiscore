# MSA Difficulty Project

## Setup
1. Use the `environment.yml` file to set up the conda environment:
    ```shell
    conda env create -f environment.yml
    ```
2. Activate the new conda environment:
    ```shell
    conda activate difficulty
    ```
3. Install all required additional tools on your system. The following are required:
   - [RAxML-NG](https://github.com/amkozlov/raxml-ng/wiki/Installation)
   - [adaptive RAxML-NG](https://github.com/amkozlov/raxml-ng/wiki/Installation#building-adaptive-branch)
   - [Muscle5](https://github.com/rcedgar/muscle/releases/tag/5.1.0)
   - [Muscle3](https://drive5.com/muscle/downloads_v3.htm)
   - [Clustal Omega](http://www.clustal.org/omega/)
   - [MAFFT](https://mafft.cbrc.jp/alignment/software/)
   
   Note that as of now (April 2024) the installation of Muscle5 did not work on my MacBook with M1 chip.
4. Adjust the `config.yaml` to set the correct paths to all executables of the above tools.

## Generating the Data
I implemented the pipeline using Snakemake. If you are unfamiliar with Snakemake, I recommend starting with the [Snakemake Tutorial](https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html) before continuing with the below instructions.

When you download/copy pre-existing data, Snakemake will typically not detect them as existing and will try to recompute everything.
Especially the MSA generation can consume a lot of time and resources, therefore I recommend running the following command in case you have existing data:

```shell
snakemake --touch --cores 1
```

Note that for the pre-existing data, I computed 16 MSAs for each tool. If you change this number, you need to recompute all data.

To execute the pipeline, run the following command:
```shell
snakemake --cores [n_cores]
```
Make sure to check how many cores you system has available. The number of cores times the number of threads used for each MSA generation (as specified in `config.yaml`) must not exceed the number of cores your system has.

Once your pipeline is all setup and running, the following additional flags might be useful:
- `-k`: Continues the execution of independent jobs in case of failing jobs. E.g. if MAFFT is unable to generate MSAs for one dataset due to some weird error, but everything else works, snakemake will continue to execute everything that does not require the results of the MAFFT MSAs. 
- `--rerun-incomplete`: In case you terminate the pipeline or an error occurs, some runs might be incomplete and file handles not properly cleaned up. In this case, snakemake will remove the affected files and start over.

See [the Snakemake documentation](https://snakemake.readthedocs.io/en/stable/executing/cli.html) for a full list of command line flags.


### Existing data
I have generated Ensembles (each with 16 MSAs) using Muscle5, Muscle3, MAFFT, Clustal Omega for about 2100 DNA MSAs from TreeBASE.
We have more DNA datasets from TreeBASE (~7000 more) and around 5000(?) from RAxML-Grove.


## Input data
The pipeline expects MSAs as input. The reason for this is that we used empirical data from TreeBASE that comes as aligned sequences.
The first step in the pipeline is the de-alignment of the MSA.
Note that at the moment, the pipeline only supports DNA sequences.

## Results
The pipeline will result in three summary files that can be used for subsequent analyses and predictor training:
- `datasets.parquet`: Features of the unaligned, raw sequences for all datasets in the input directory. This dataframe will contain one row per dataset.
- `ensembles.parquet`: Features of the ensembles of MSAs generated for all datasets in the input directory. This dataframe will contain multiple rows per dataset, one row for each tool that was used to generate a set of MSAs, and one row for the features of all MSAs generated across all tools. 
- `trees.parquet`: Features of RAxML-NG tree inferences for each dataset and each ensemble. This dataframe will contain multiple rows per dataset, one row for each tool that was used to generate a set of MSAs, and one row for the features of all MSAs generated across all tools. 

For each dataset, Snakemake generates one directory (named similar to the input dataset) containing:
```text
{dataset}
├── all_tools.efa  # All generated MSA replicates for all tools as Muscle5 .efa file. 
├── all_tools.raxmlng.bestTrees  # All best RAxML-NG trees for each MSA for each tool.
├── {clustalo/mafft/muscle3/muscle5}  # Note that the naming of the individual MSAs in muscle5 differs
│         ├── efa_explode.log 
│         ├── ensemble.efa  # All generated MSA replicates for this tool as Muscle5 .efa file
│         ├── ensemble.log  # Logfile of the ensemble generation
│         ├── ensemble.pckl  # Pickle of the Ensemble python object 
│         ├── msa.{i}.fasta  # for i in range(msas_per_ensemble)
│         ├── raxmlng  # RAxML-NG tree inference results for each MSA
│         │         ├── msa.{i}.msa.raxml.bestModel
│         │         ├── msa.{i}.msa.raxml.bestTree
│         │         ├── msa.{i}.msa.raxml.log
│         │         ├── msa.{i}.msa.raxml.mlTrees
│         │         ├── msa.{i}.msa.raxml.rba
│         │         ├── msa.{i}.msa.raxml.startTree
│         ├── raxmlng.bestTrees  # Best RAxML-NG tree for each MSA
│         ├── raxmlng.log  # Log file containing the results of the RF-distance RAxML-NG run for raxmlng.bestTrees
├── dataset_features.parquet  # Parquet file containing the pandas dataframe of Dataset features for this dataset. See features/features.md for a detailed description of the features.
├── ensemble_features.parquet  # Parquet file containing the pandas dataframe of Ensemble features for this dataset. See features/features.md for a detailed description of the features and see below for details on the rows of this dataframe.
├── raxmlng_tree_features.parquet  # RF-Distance results for each tool and all tools combined. See below for details.
└── sequences.fasta  # Unaligned sequences
```

`ensemble_features.parquet` and`raxmlng_tree_features.parquet`: 
Both dataframes contain one row for each tool and one row for all tools combined as indicated by the `"tool"` column. The row with `"tool" == "all""` corresponds to the results for all MSA results of all tools combined. 