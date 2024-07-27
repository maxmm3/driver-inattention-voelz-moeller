preproc.ipynb - used for image processing and data augmentation, should be executed first.
load.py - was used to normalize data in seperate script. Originated from times when software crashed from too large amounts of data.
alt_method.ipynb - requires pip install inference_sdk to use the roboflow api and was used to evaluate the alternative model.
cross_val_final_model.py - was used for cross final cross-validation.
cross_val_preproc.ipynb - was used to create the cross validation data sets.
eval.ipynb - was used for some plotting and evaluation of the models performance.
final_model.py - is the file for training the optimized model on the original data.
main.py - contains the unoptimized model and underwent some changes as it was used for trying things out.
parameter_tuning.py - was used to do the hyperparameter search.

All data was stored in a folder called data/ lying next to the scripts/ folder.