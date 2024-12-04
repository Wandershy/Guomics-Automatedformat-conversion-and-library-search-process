# Scripts for "Guomics-Automatedformat-conversion-and-library-search-process"

```
python3 AutoSearch_wsy_20241202.py -p /storage/guotiannanLab/xingziyuan/01.metaMS-GPT/00.training_dataset/01.single.species/00.bacteria/00.test/ -i species.txt
```

- -p pathway of your species files
- -i species list that you want to process

> **Note** Each species file should include a `.fasta` file located in the `/pathway/species/` directory and a `pride_raw` folder within the same directory. Additionally, the raw data should be stored in the `PXD*` folder within the `/pathway/species/pride_raw/` directory.
> 
