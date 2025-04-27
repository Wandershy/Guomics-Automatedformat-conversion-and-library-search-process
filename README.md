# Scripts for "Guomics-Automatedformat-conversion-and-library-search-process"


> **Tips!** 
>
>Before running the script, you need to create a conda environment named `sage` and ensure that you have the necessary permissions to access the `/home/guotiannanLab/xingziyuan/` directory.

### Please follow this step to install Sage:

~~~
# bash

# create a conda environment named `sage` and activate it
conda create -n sage
conda activate sage

# install sage package from bioconda
conda install -c bioconda -c conda-forge sage-proteomics
~~~

### Run AutoSearch

```
python3 AutoSearch_final_WangSY20241208.py -p /storage/guotiannanLab/xingziyuan/01.metaMS-GPT/00.training_dataset/01.single.species/00.bacteria/00.test/ -i species.txt -t raw
```

- -p pathway of your species files
- -i species list that you want to process
- -t input rawdata type,choices=["raw", "d"], default raw

> **Note** Each species file should include a `.fasta` file located in the `/pathway/species/` directory and a `pride_raw` folder within the same directory. Additionally, the raw data should be stored in the `PXD*` folder within the `/pathway/species/pride_raw/` directory.

>Each species directory will produce three `.txt` files (`/pathway/species/failed_msConvert_files.txt`, `/pathway/species/failed_Sage_files.txt`, `/pathway/species/failed_Fragpipe_files.txt`), They counted the names of unconverted and unsearched files respectively. `-p` will also generate a `statistics.txt` file, with each column representing `Species`, `# projects`, `# files`, `fasta_name`, `# files completed msConvert`, `# files completed Sage` and `# files completed FragPipe`
