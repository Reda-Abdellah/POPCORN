from training import POPCORN

POPCORN( ps=[64,64,64], dataset_path="../", Epoch_per_step=4, increment=200, datafolder='dataset/data_nearest/',datafolder_val='dataset/data_nearest_val/',dataselection_strategy='nearest',
                resume=True, resume_after_adding_pseudo_of_step=8, load_precomputed_features=False, unlabeled_dataset="OFSEP_and_volbrain",recompute_distance_each_step=False,
                  regularized=True, loss_weights=[1,0.01], k=5)