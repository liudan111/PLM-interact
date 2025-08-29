Data
====

Trained Models
--------------
Human PPI models trained on the cross-species dataset
- `danliu1226/PLM-interact-650M-humanV11 <https://huggingface.co/danliu1226/PLM-interact-650M-humanV11>`_ 
- `danliu1226/PLM-interact-35M-humanV11 <https://huggingface.co/danliu1226/PLM-interact-35M-humanV11>`_

Human PPI model trained on the Bernett dataset
- `danliu1226/PLM-interact-650M-Leakage-Free-Dataset <https://huggingface.co/danliu1226/PLM-interact-650M-Leakage-Free-Dataset>`_ 

Human PPI model trained on the STRING V12 training dataset
- `danliu1226/PLM-interact-650M-humanV12 <https://huggingface.co/danliu1226/PLM-interact-650M-humanV12>`_ 

Virus-human PPI model
- `danliu1226/PLM-interact-650M-VH <https://huggingface.co/danliu1226/PLM-interact-650M-VH>`_ 

Mutation effect model 
- `danliu1226/PLM-interact-650M-Mutation <https://huggingface.co/danliu1226/PLM-interact-650M-Mutation>`_ 


Pre-trained models can be downloaded from `Hugging Face <https://huggingface.co/danliu1226>`_.

.. code-block:: bash
    from huggingface_hub import snapshot_download
    import os
    # The ID of the repository you want to download
    repo_id = "danliu1226/PLM-interact-650M-humanV11" # Or any other repo
    # The local directory where you want to save the folder
    local_dir = "../offline/PLM-interact-650M-humanV11"
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading repository '{repo_id}' to '{local_dir}'...")

    # Use snapshot_download with force_download=True
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        force_download=True  # <-- ADD THIS LINE
    )
    print("\nDownload complete!")
    print(f"All files for {repo_id} are saved in the '{local_dir}' folder.")


Datasets for PLM-interact
--------------------------
- `Cross-species dataset: https://huggingface.co/datasets/danliu1226/cross_species_benchmarking`_
- `Bernett dataset: https://huggingface.co/datasets/danliu1226/Bernett_benchmarking`_
- `Mutation effect dataset: https://huggingface.co/datasets/danliu1226/Mutation_effect_dataset`_
- `Virus-human PPI dataset: https://huggingface.co/datasets/danliu1226/virus_human_benchmarking`_
- `STRING V12 training dataset: https://huggingface.co/datasets/danliu1226/STRING_V12_TrainingSet`_


Original Datasets
-----------
- `Cross-species dataset: https://github.com/samsledje/D-SCRIPT/blob/main/data/`_[1]
- `Bernett dataset: https://doi.org/10.6084/m9.figshare.21591618.v3`_[2]
- `Mutation effect dataset:https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv`_[3]
- `Virus-human PPI dataset: http://kurata35.bio.kyutech.ac.jp/LSTM-PHV/download_page`_[4]
- `STRING V12 training dataset: https://stringdb-downloads.org/download/protein.physical.links.v12.0.txt.gz`_[5]

References
-----------
[1] Sledzieski, S., Singh, R., Cowen, L. & Berger, B. D-SCRIPT translates genome to phenome with sequence-based, structure-aware, genome-scale predictions of protein-protein interactions. Cell Systems 12, 969-982.e6 (2021).

[2] BBernett, J., Blumenthal, D. B. & List, M. Cracking the black box of deep sequence-based protein–protein interaction prediction. Briefings in Bioinformatics 25, bbae076 (2024).

[3] Kerrien, S. et al. The IntAct molecular interaction database in 2012. Nucleic Acids Research 40, D841–D846 (2012).

[4]Tsukiyama, S., Hasan, M. M., Fujii, S. & Kurata, H. LSTM-PHV: prediction of human-virus protein–protein interactions by LSTM with word2vec. Briefings in Bioinformatics 22, bbab228 (2021).

[5] Szklarczyk, D. et al. The STRING database in 2023: protein–protein association networks and functional enrichment analyses for any sequenced genome of interest. Nucleic Acids Research 51, D638–D646 (2023).