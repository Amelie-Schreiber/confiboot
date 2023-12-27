# (Unofficial Implementation) DiffDock-CB: The Discovery of Binding Modes Requires Rethinking Docking Generalization

[The Discovery of Binding Modes Requires Rethinking Docking Generalization](https://openreview.net/forum?id=FhFglOZbtZ).

Here we provide our implementation of the Confidence Bootstrapping method, pretrained diffusion and confidence models, and processed receptors from the DockGen-clusters validation and test sets. 

---

## Alternate Setup:


*Setting up the conda/mamba environment using the `environment.yml` file has been problematic for me, so I decided to try setting up the environment manually using [DiffDock-Pocket](https://anonymous.4open.science/r/DiffDock-Pocket-AQ32/README.md) as a template. This seems to work, but there may still be bugs. Try creating an environment using the `environment_alternate_2.yml` file as follows:*

**Clone this Repo:**
```
git clone https://github.com/Amelie-Schreiber/confiboot.git
```
**Change Directories:**
```
cd confiboot
```
**Create Conda Environment:**
```
conda env create -f environment_alternate_2.yml
```
or if you are using mamba
```
mamba env create -f environment_alternate_2.yml
```
**Activate the Environment:**
```
conda activate diffdock-cb1
```
Now you can skip the "**Setup**" section below, move on to the "**ESM Embeddings**" section, and just use `diffdock-cb1` as your environment. 

---

## Dataset

The Binding MOAD database can be downloaded from [http://www.bindingmoad.org/]. All 189 complexes in the DockGen test set can be found in `data/BindingMOAD_2020_ab_processed_biounit/test_names.npy`, and the 85 complexes from 8 clusters we tested Confidence Bootstrapping on can be found in `data/BindingMOAD_2020_ab_processed_biounit/test_names_bootstrapping.npy`. The list of complexes in the DockGen benchmark can be found at `data/BindingMOAD_2020_ab_processed_biounit/new_cluster_to_ligands.pkl`. Complexes from Binding MOAD should be downloaded also to `data/BindingMOAD_2020_ab_processed_biounit`. Here, we also provide the ligands and receptors used in the test set at `data/BindingMOAD_2020_ab_processed_biounit/MOAD_ligands` and `data/MOAD_new_test_processed` respectively.

## Setup

We will set up the environment with anaconda [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), and have provided an `environment.yml` file. While in the project directory, run
    
    conda env create

Activate the environment

    conda activate diffdock

## ESM Embeddings

In order to run the diffusion model, we need to generate ESM2 embeddings for complexes in Binding MOAD. First we prepare sequences:

    python datasets/moad_lm_embedding_preparation.py --data_dir data/MOAD_new_test_processed

Then, we install esm and generate embeddings for the test proteins:
    
    git clone https://github.com/facebookresearch/esm
    cd esm
    pip install -e .
    cd ..
    HOME=esm/model_weights python esm/scripts/extract.py esm2_t33_650M_UR50D data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.fasta data/esm2_output --repr_layers 33 --include per_tok
    
Then we convert the embeddings to a single `.pt` file:

    python datasets/esm_embeddings_to_pt.py --esm_embeddings_path data/esm2_output

    

## Running finetuning:

After downloading the complexes from Binding MOAD, we can run the Confidence Bootstrapping finetuning on a cluster like `Homo-oligomeric flavin-containing Cys decarboxylases, HFCD`:

    python -m finetune_train --test_sigma_intervals --log_dir workdir --lr 1e-4 --batch_size 5 --ns 32 --nv 6 --scale_by_sigma --dropout 0.1 --sampling_alpha 2 --sampling_beta 1 --remove_hs --c_alpha_max_neighbors 24 --cudnn_benchmark --rot_alpha 1 --rot_beta 1 --tor_alpha 1 --tor_beta 1 --n_epochs 300 --sh_lmax 1 --num_prot_emb_layers 3 --reduce_pseudoscalars --moad_esm_embeddings_sequences_path data/BindingMOAD_2020_ab_processed_biounit/sequences_to_id.fasta --moad_esm_embeddings_path data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.pt --enforce_timesplit --moad_dir data/BindingMOAD_2020_ab_processed_biounit --dataset distillation --inference_out_dir results/triple_crop20 --confidence_cutoff -4 --restart_ckpt best_ema_inference_epoch_model --restart_dir workdir/pretrained_score_model --filtering_model_dir workdir/pretrained_confidence_model --max_complexes_per_couple 20 --distillation_train_cluster "Homo-oligomeric flavin-containing Cys decarboxylases, HFCD" --fixed_length 100 --initial_iterations 10 --minimum_t 0.3 --cache_path cache --inference_batch_size 4 --save_model_freq 5 --inference_iterations 4 --val_inference_freq 5 --inference_samples 8 --split test 

or: 
```
python -m finetune_train \
  --test_sigma_intervals \
  --log_dir workdir \
  --lr 1e-4 \
  --batch_size 5 \
  --ns 32 \
  --nv 6 \
  --scale_by_sigma \
  --dropout 0.1 \
  --sampling_alpha 2 \
  --sampling_beta 1 \
  --remove_hs \
  --c_alpha_max_neighbors 24 \
  --cudnn_benchmark \
  --rot_alpha 1 \
  --rot_beta 1 \
  --tor_alpha 1 \
  --tor_beta 1 \
  --n_epochs 300 \
  --sh_lmax 1 \
  --num_prot_emb_layers 3 \
  --reduce_pseudoscalars \
  --moad_esm_embeddings_sequences_path data/BindingMOAD_2020_ab_processed_biounit/sequences_to_id.fasta \
  --moad_esm_embeddings_path data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.pt \
  --enforce_timesplit \
  --moad_dir data/BindingMOAD_2020_ab_processed_biounit \
  --dataset distillation \
  --inference_out_dir results/triple_crop20 \
  --confidence_cutoff -4 \
  --restart_ckpt best_ema_inference_epoch_model \
  --restart_dir workdir/pretrained_score_model \
  --filtering_model_dir workdir/pretrained_confidence_model \
  --max_complexes_per_couple 20 \
  --distillation_train_cluster "Homo-oligomeric flavin-containing Cys decarboxylases, HFCD" \
  --fixed_length 100 \
  --initial_iterations 10 \
  --minimum_t 0.3 \
  --cache_path cache \
  --inference_batch_size 4 \
  --save_model_freq 5 \
  --inference_iterations 4 \
  --val_inference_freq 5 \
  --inference_samples 8 \
  --split test
```

Note that the command above is not the same as the one used in experiments in the paper, which also samples random complexes from PDBBind at every bootstrapping step. To reproduce paper results, we need to download the PDBBind dataset:

    1. download it from [zenodo](https://zenodo.org/record/6034088) 
    2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`

Then, we can run the finetuning command with `--keep_original_train` and `--totoal_trainset_size 100` to reproduce paper numbers.

## Inference

Try modifying the following command to run inference:

```
python script.py \
  --config <CONFIG_FILE> \
  --model_dir workdir \
  --ckpt best_model.pt \
  --filtering_model_dir <FILTERING_MODEL_DIR> \
  --filtering_ckpt best_model.pt \
  --affinity_model_dir <AFFINITY_MODEL_DIR> \
  --affinity_ckpt best_model.pt \
  --num_cpu <NUM_CPU> \
  --run_name test \
  --project ligbind_inf \
  --out_dir <OUT_DIR> \
  --batch_size 40 \
  --old_score_model \
  --old_filtering_model \
  --old_affinity_model \
  --matching_popsize 40 \
  --matching_maxiter 40 \
  --esm_embeddings_path <ESM_EMBEDDINGS_PATH> \
  --moad_esm_embeddings_sequences_path <MOAD_ESM_EMBEDDINGS_SEQUENCES_PATH> \
  --chain_cutoff <CHAIN_CUTOFF> \
  --use_full_size_protein_file \
  --use_original_protein_file \
  --save_complexes \
  --complexes_save_path <COMPLEXES_SAVE_PATH> \
  --dataset moad \
  --cache_path data/cacheMOAD \
  --data_dir data/BindingMOAD_2020_ab_processed_biounit/ \
  --split_path data/BindingMOAD_2020_ab_processed/splits/val.txt \
  --no_model \
  --no_random \
  --no_final_step_noise \
  --overwrite_no_final_step_noise \
  --ode \
  --wandb \
  --overwrite_wandb \
  --inference_steps 20 \
  --limit_complexes <LIMIT_COMPLEXES> \
  --num_workers 1 \
  --tqdm \
  --save_visualisation \
  --samples_per_complex 4 \
  --resample_rdkit \
  --skip_matching \
  --sigma_schedule expbeta \
  --inf_sched_alpha 1 \
  --inf_sched_beta 1 \
  --different_schedules \
  --overwrite_different_schedules \
  --rot_sigma_schedule expbeta \
  --rot_inf_sched_alpha 1 \
  --rot_inf_sched_beta 1 \
  --tor_sigma_schedule expbeta \
  --tor_inf_sched_alpha 1 \
  --tor_inf_sched_beta 1 \
  --pocket_knowledge \
  --no_random_pocket \
  --overwrite_pocket_knowledge \
  --pocket_tr_max 3 \
  --pocket_cutoff 5 \
  --actual_steps <ACTUAL_STEPS> \
  --xtb \
  --use_true_pivot \
  --restrict_cpu \
  --force_fixed_center_conv \
  --protein_file protein_processed \
  --unroll_clusters \
  --remove_pdbbind \
  --split val \
  --limit_failures 5 \
  --min_ligand_size <MIN_LIGAND_SIZE> \
  --max_receptor_size <MAX_RECEPTOR_SIZE> \
  --remove_promiscuous_targets <REMOVE_PROMISCUOUS_TARGETS> \
  --svgd_weight_log_0 <SVG_WEIGHT_LOG_0> \
  --svgd_weight_log_1 <SVG_WEIGHT_LOG_1> \
  --svgd_repulsive_weight_log_0 <SVG_REPULSIVE_WEIGHT_LOG_0> \
  --svgd_repulsive_weight_log_1 <SVG_REPULSIVE_WEIGHT_LOG_1> \
  --svgd_langevin_weight_log_0 <SVG_LANGEVIN_WEIGHT_LOG_0> \
  --svgd_langevin_weight_log_1 <SVG_LANGEVIN_WEIGHT_LOG_1> \
  --svgd_kernel_size_log_0 <SVG_KERNEL_SIZE_LOG_0> \
  --svgd_kernel_size_log_1 <SVG_KERNEL_SIZE_LOG_1> \
  --svgd_rot_log_rel_weight <SVG_ROT_LOG_REL_WEIGHT> \
  --svgd_tor_log_rel_weight <SVG_TOR_LOG_REL_WEIGHT> \
  --svgd_use_x0 \
  --temp_sampling_tr 1.0 \
  --temp_psi_tr 0.0 \
  --temp_sampling_rot 1.0 \
  --temp_psi_rot 0.0 \
  --temp_sampling_tor 1.0 \
  --temp_psi_tor 0.0 \
  --temp_sigma_data 0.5 \
  --gnina_minimize \
  --gnina_log_file gnina_log.txt \
  --gnina_full_dock \
  --save_gnina_metrics \
  --gnina_autobox_add 4.0 \
  --gnina_poses_to_optimize 1
```
