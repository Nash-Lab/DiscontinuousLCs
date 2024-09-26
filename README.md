 # Data-Error Scaling in Machine Learning on Natural Discrete Combinatorial Mutation-prone Sets: Case Studies on Peptides and Small Molecules

 Data, Python scripts and Results (available free of charge on DOI:[10.5281/zenodo.11148307](https://zenodo.org/doi/10.5281/zenodo.11148307)) used in the publication entitled "Data-Error Scaling in Machine Learning on Natural Discrete Combinatorial Mutation-prone Sets: Case Studies on Peptides and Small Molecules" (--> [preprint](https://arxiv.org/abs/2405.05167)).

 - [Overview](##Overview)
 - [Requirements](##Requirements)
 - [Citation](##Citation)

 The files are generally separated based on the graph topology (molecules / peptides).

 ## Overview

Abstract: We investigate trends in the data-error scaling behavior of machine learning (ML) models trained on discrete combinatorial spaces that are prone-to-mutation, such as proteins or organic small molecules. We trained and evaluated kernel ridge regression machines using variable amounts of computationally generated training data. Our synthetic datasets comprise i) two naïve functions based on many-body theory; ii) binding energy estimates between a protein and a mutagenised peptide; and iii) solvation energies of two 6-heavy atom structural graphs. In contrast to typical data-error scaling, our results showed discontinuous monotonic phase transitions during learning, observed as rapid drops in the test error at particular thresholds of training data. We observed two learning regimes, which we call saturated and asymptotic decay, and found that they are conditioned by the level of complexity (i.e. number of mutations) enclosed in the training set. We show that during training on this class of problems, the predictions were clustered by the ML models employed in the calibration plots. Furthermore, we present  an alternative strategy to normalize learning curves (LCs) and the concept of mutant based shuffling. This work has implications for machine learning on mutagenisable discrete spaces such as chemical properties or protein phenotype prediction, and improves basic understanding of concepts in statistical learning theory. 


![](F1.png)
*Workflow overview. (A) Database Generation: a table containing all possible mutagenized peptide variants was generated from a starting construct (WT) and a mutational vocabulary. The response variable (binding energy) was computed for each entry. (B) Encoding: the database was converted into a matrix containing numerical values using binary flattened one hot encoding. (C) Machine learning: Laplacian kernel machines were trained using different quantities of data and different shuffling strategies. (D) Evaluation: LCs and calibration plots were used to study the learning process. The amount of information used during training in the scatter plots is reported on the figure.*
 ### Data (zipped and on Zenodo)
 This folder contains the raw data used during this work.
 `out_seq_total.txt` contains information on the sequences used (mutations, number of mutations, etc.). `output_energies_total.txt` contains the response variables, which include: a) unrelaxed EvoEF energies (peptides); b) relaxed EvoEF energies (peptides, `*_repaired.txt`); and c) solvation energies (molecules).
 3D structures are provided in `.xyz` format in the subfolder `XYZ` (molecules).

#### Note on Data location
In order to properly work, one should allocate the used script (`Scripts/script_synthetic_peptide.py` or `Scripts/script_chemSpace.py`) and the appropriate datasets (`Data/peptides/*` or `Data/molecules/*`) in the same subpath.

 ### Scripts
 This folder contains the scripts used during this work to train and save the ML models. Different options are available to modify the behaviour of the scripts. For further information please use the following command:

 ```
python3 ./script.py -h
```

 To test both scripts, one could run the following two lines (requirements needed, see below). The columns of the expected outputs correspond to: 1) number of replicate; 2) N_{train}^{norm}; 3) N_{train}; 4) Prediction error; 5) time limit (important only for SLURM); and 6) Time elapsed.

For Chemical Space part:
  ```
python3 -u script_chemSpace.py -i 0. 1. 3. 4. 4. 5.
```

Expected output (might be slightly different):
  ```
0 0.0 2 16.036373515625 inf 1.7253291606903076
0 0.05 2 16.036373515625 inf 1.7522239685058594
0 0.1 3 13.993278475547983 inf 1.780350923538208
0 0.15000000000000002 3 13.993278475547983 inf 1.808826208114624
0 0.2 4 10.896933740798525 inf 1.8383302688598633
0 0.25 4 10.896933740798525 inf 1.8679602146148682
0 0.30000000000000004 5 10.758454312031157 inf 1.8956100940704346
0 0.35000000000000003 6 8.365517165893348 inf 1.9245600700378418
0 0.4 6 8.365517165893348 inf 1.952970027923584
0 0.45 7 6.993938353546028 inf 1.9835031032562256
0 0.5 8 5.966052924467804 inf 2.01668119430542
0 0.55 9 4.629995124133443 inf 2.0553152561187744
0 0.6000000000000001 11 3.238898320292327 inf 2.096766233444214
0 0.65 12 3.37409678457585 inf 2.1398301124572754
0 0.7000000000000001 13 3.3971114823610074 inf 2.1801059246063232
0 0.75 15 3.40079058612884 inf 2.221668243408203
0 0.8 17 3.4273935739808565 inf 2.265669107437134
0 0.8500000000000001 19 3.3822941387870897 inf 2.313124179840088
0 0.9 21 3.478512091282942 inf 2.3630521297454834
0 0.9500000000000001 23 3.4293628156427616 inf 2.416574239730835
0 1.0 26 2.8087297207090223 inf 2.4814462661743164
```

For Synthetic Peptide part:
  ```
python3 -u script_synthetic_peptide.py -i 0. 1. 3. 4. 4. 5.
```

Expected output (might be slightly different):
  ```
0 0.0 1 3.9869511968181284 inf 22.303447008132935
0 0.05 1 3.9869511968181284 inf 22.408642053604126
0 0.1 1 3.9869511968181284 inf 22.51335883140564
0 0.15000000000000002 2 3.3833400960289053 inf 22.73532199859619
0 0.2 2 3.3833400960289053 inf 22.974663019180298
0 0.25 3 3.367568701821707 inf 23.302359104156494
0 0.30000000000000004 3 3.367568701821707 inf 23.570611000061035
0 0.35000000000000003 4 2.950069719644046 inf 23.890959978103638
0 0.4 4 2.950069719644046 inf 24.223950147628784
0 0.45 5 2.939314332397317 inf 24.605826139450073
0 0.5 6 2.898444798370211 inf 25.02949619293213
0 0.55 7 2.806807002881295 inf 25.62370014190674
0 0.6000000000000001 9 2.7332493082753406 inf 26.287458181381226
0 0.65 10 2.6777094975193387 inf 27.079219102859497
0 0.7000000000000001 12 2.604500636754286 inf 28.004189014434814
0 0.75 14 2.529645789681478 inf 29.289401054382324
0 0.8 16 2.4479772295628246 inf 30.76276206970215
0 0.8500000000000001 19 2.0927477301734405 inf 32.40441703796387
0 0.9 22 2.0150156790473956 inf 34.575512170791626
0 0.9500000000000001 25 2.0044781872687745 inf 37.10925507545471
0 1.0 29 1.532384570988287 inf 39.80328607559204
```

 ### Results (on Zenodo)
 This folder (available on DOI:[10.5281/zenodo.11148307](https://zenodo.org/doi/10.5281/zenodo.11148307)) contains the results (outputs) of the ML models trained using the provided scripts (see previous subsection). Such results are incuded in the form of `.npy` files. To load the files please include the option `allow_pickle=True`.

 Each `.npy` file contains the following keys:
 * `initial_parameters`: script inputs.
 * `d_encoder`: encoder used (not always included).
 * `ns_train`: number of training points used for the LCs (rounded, integers).
 * `ns_train_float`: number of training points used for the LCs (not rounded, float).
 * `ns_train_norm`: number of training points used for the LCs (normalized, float).
 * `res`: test MAEs.
 * `res_tot`: (train,validation,test) MAEs.
 * `res_tot_mut`: (train,validation,test) MAEs sorted by mutation number.
 * `l_opt`: optimal kernel length used during the test.
 * `ls`: kernel lengths used for grid search.
 * `idx_seeds`: indices used to reshuffle the data. If one want to rebild the initial order use `np.argsort(idx_seeds)`.
 * `alpha_opt`: optimal regression parameters used to calculate the test error. To sort the data use `alpha_opt[i][ii][np.argsort(idx_seeds[ii,0:arg_train_max].astype(int)[:ns_train[i]]`. Where `i` is the replicate number (0-99) and ii is the idex in the LC.
 * `valid_errs`: validation error (MAE) calculated for each point in the hyperparameter (kernel scale) optimisation.
 * `test_errs`: test error (MAE) calculated for each point in the hyperparameter (kernel scale) optimisation.

By default, results are not exported when using a Darwin-based computer (MacOS). If needed, this behaviour can be easily changed by modifying the last few lines of code in each script.

 The test examples (see previous subsection) are included locally (here) and on zenodo.

## Requirements
 The scripts has been run locally on and Intel-based Macbook Pro (Macos 13.6.6) and distributed in a CentOS-SLURM based HPC cluster ([scicore website](https://scicore.unibas.ch)).

 To use our scripts, one should install the libraries in `environment.yml`.

```
conda env create --file=environment.yml
```
### Note on qmlcode
Some options (Coulomb Matrix based encoding) available in `Scripts/script_chemSpace.py`, require additionally to install the python library `qmlcode` ([website](http://qmlcode.org), not included in the `environment.yml`). To do that, one can follow the installation instruction provided on the qmlcode webpage ([installation guide](http://www.qmlcode.org/installation.html#)).


## Citation
For usage of the code or data, and associated manuscript, please cite according to the enclosed [citation.bib](citation.bib).
