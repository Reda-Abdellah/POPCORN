from training import POPCORN

os.environ["CUDA_VISIBLE_DEVICES"]='0'

POPCORN( ps=[64,64,64], dataset_path="/lib/", Epoch_per_step=2, increment_new_data=200, datafolder='data_nearest/',dataselection_strategy='random',
                resume=False, resume_after_adding_pseudo_of_step=1, load_precomputed_features=False, unlabeled_dataset="volbrain",recompute_distance_each_step=False,
                  regularized=False, loss_weights=[1,0.01], k=5)