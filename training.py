import os, glob, sys
import numpy as np
import nibabel as nii
import modelos
from utils import *
import losses
import torch
from helper import *
import torch.optim as optim

def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)

def POPCORN( ps=[64,64,64], dataset_path="/lib/", Epoch_per_step=2, increment_new_data=200, datafolder='data_nearest/',dataselection_strategy='nearest',modality="FLAIR",
                resume=False, resume_after_adding_pseudo_of_step=1, load_precomputed_features=False, unlabeled_dataset="volbrain",recompute_distance_each_step=True,
                  regularized=True, loss_weights=[1,0.01], k=5):

    
    if(regularized):
        in_filepath='weights/SUPERVISED_regularized.h5'
    else:
        in_filepath='weights/SUPERVISED_noreg.h5'
    
    model=torch.load(in_filepath)
    
    if(unlabeled_dataset=="volbrain"):
        listaFLAIR = sorted(glob.glob(dataset_path+"/volbrain_qc/n_mfmni*flair*.nii*"))
        listaMASK = sorted(glob.glob(dataset_path+"/volbrain_qc/mask*.nii*"))
        listaMASK = np.array(listaMASK)
    elif(unlabeled_dataset=="OFSEP_and_volbrain"):
        listaFLAIR = sorted(glob.glob(dataset_path+"/volbrain_qc/n_mfmni*flair*.nii*"))+sorted(glob.glob(dataset_path+"/OFSEP/data/n_mfmni*flair*.nii*"))
        listaMASK = sorted(glob.glob(dataset_path+"/volbrain_qc/mask*.nii*"))+sorted(glob.glob(dataset_path+"/OFSEP/data/mask*.nii*"))
        listaMASK = np.array(listaMASK)
    elif(unlabeled_dataset=="isbi_test"):
        listaFLAIR = sorted(glob.glob(dataset_path+"/ISBI_preprocess/test*flair*.nii*"))

    listaFLAIR=np.array(listaFLAIR)

    lib_path_3 = os.path.join(dataset_path,"isbi_final_train_preprocessed")

    listaFLAIR_3=keyword_toList(path=lib_path_3,keyword="flair")
    listaSEG1_3=keyword_toList(path=lib_path_3,keyword="mask1")
    listaSEG2_3=keyword_toList(path=lib_path_3,keyword="mask2")

    listaFLAIR_labeled= np.array(listaFLAIR_3)

    unlabeled_indxs= range(len(listaFLAIR))
    pseudolabeled_indxs=[]
    unlabeled_num=len(unlabeled_indxs)
    pseudolabeled_num=len(pseudolabeled_indxs)
    labeled_num=len(listaFLAIR_labeled)

    update_labeled_folder_flair(listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)
    update_labeled_folder_flair(listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,numbernotnullpatch=15)


    step=0
    model.load_weights(in_filepath)

    if( (not recompute_distance_each_step) and dataselection_strategy=='nearest'):
        if(load_precomputed_features):
            print('loading precomputed features...')
            rank_distance=np.load('rank_distance_volbrain_tsne3_regularized.npy')
        else:
            print('computing bottleneck_features...')
            bottleneck_features_labeled,file_names= features_from_names_flair(listaFLAIR_labeled,fun)
            bottleneck_features_unlabeled,file_names= features_from_names_flair(listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
            print('computing projection to simplar dataplane...')
            rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
            np.save('rank_distance_volbrain_tsne3_regularized.npy',rank_distance)

    if(resume):
        for it in range(resume_after_adding_pseudo_of_step):
            print('resuming training...')
            if(dataselection_strategy=='nearest'):
                new_pos_in_features = give_dist_for_Kclosest(rank_distance,n_indxs=increment_new_data,k=k)
            elif(dataselection_strategy=='random'):
                np.random.seed(43+it+1)
                new_pseudo = np.array(unlabeled_indxs)
                np.random.shuffle(new_pseudo)
                new_pseudo =new_pseudo[:increment_new_data]
                new_pseudo=new_pseudo.tolist()
                new_pos_in_features= new_pseudo
            not_new_pos_in_features = [x for x in range(unlabeled_num) if x not in new_pos_in_features]
            #update indexes
            pseudolabeled_indxs= pseudolabeled_indxs+ new_pos_in_features
            unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]
            #update num
            unlabeled_num=len(unlabeled_indxs)
        step=resume_after_adding_pseudo_of_step-1
        if(not step==0):
            model.load_weights(out_filepath(step))


    #Training
    while(unlabeled_num>increment_new_data):
        step=step+1
        out_filepath= 'weights/'+sys.argv[0].replace('.py','_step')+"%02d" % (x)+'.h5'
        print('step: '+str(step))
        print('loading new data...')
        if( resume and step==resume_after_adding_pseudo_of_step):
            print('resuming..')
        else:
            
            if(dataselection_strategy=='nearest'):
                if(recompute_distance_each_step):
                    bottleneck_features_labeled,file_names= features_from_names_flair(listaFLAIR_labeled,fun)
                    bottleneck_features_unlabeled,file_names= features_from_names_flair(listaFLAIR[unlabeled_indxs],fun,listaMASK[unlabeled_indxs])
                    if(len(pseudolabeled_indxs)>0):
                        bottleneck_features_pseudolabeled,file_names= features_from_names_flair(listaFLAIR[pseudolabeled_indxs],fun,listaMASK[pseudolabeled_indxs])
                        bottleneck_features_labeled=  np.concatenate((bottleneck_features_labeled,bottleneck_features_pseudolabeled),axis=0)
                    print('computing projection to simplar dataplane...')
                    rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
                    np.save('step'+str(step)+'.npy',rank_distance)
                new_pos_in_features = give_dist_for_Kclosest(rank_distance,n_indxs=increment_new_data,k=k)
            elif(dataselection_strategy=='random'):
                np.random.seed(43+it+1)
                new_pseudo = np.array(unlabeled_indxs)
                np.random.shuffle(new_pseudo)
                new_pseudo =new_pseudo[:increment_new_data]
                new_pseudo=new_pseudo.tolist()
                new_pos_in_features= new_pseudo
            print(new_pos_in_features)
            not_new_pos_in_features = [x for x in range(unlabeled_num) if x not in new_pos_in_features]
            pseudolabeled_indxs= pseudolabeled_indxs+ new_pos_in_features
            unlabeled_indxs =  [x for x in unlabeled_indxs if x not in pseudolabeled_indxs]
            #update num
            unlabeled_num=len(unlabeled_indxs)
            update_data_folder_flair(model,new_pos_in_features,listaFLAIR,listaMASK,datafolder=datafolder,regularized=regularized)
        train_files_bytiles=[]
        for i in range(27):
            train_files_bytiles.append(keyword_toList(datafolder,"x*tile_"+str(i)+".npy") )


        print('training with new data...')
        
        numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
        if(regularized):
            result=model.fit_generator(data_gen_iqda_2it(datafolder,train_files_bytiles,sim='output_diff'), 
                steps_per_epoch=numb_data,
                epochs=Epoch_per_step)
        else:
            result=model.fit_generator(data_gen_iqda(datafolder),
                steps_per_epoch=numb_data,
                epochs=Epoch_per_step)

        model.save_weights(out_filepath(step))


def train_on_labeled_only(regularized=True,save=True,  loss_weights=[1,0.01],batch_size=1 ,early_stop_treshold=30, da_type="iqda_v2",modality='T1_FLAIR', nbNN=[5,5,5],ps=[64,64,64],Snapshot=True):
    

    if(save):
        datafolder='dataset/train/'
        datafolder_val='dataset/val/'
        Epoch=100
        lib_path = os.path.join("..","all_ms_preprocessed")
        
        listaT1_1=keyword_toList(path=lib_path,keyword="mso*mprage.")
        listaFLAIR_1=keyword_toList(path=lib_path,keyword="mso*flair")
        listaT1_2=keyword_toList(path=lib_path,keyword="msseg*mprage.")
        listaFLAIR_2=keyword_toList(path=lib_path,keyword="msseg*flair")
        listaSEG_2=keyword_toList(path=lib_path,keyword="msseg*mask1")
        listaSEG_1=keyword_toList(path=lib_path,keyword="mso_mask1")
        listaT1_3=keyword_toList(path=lib_path,keyword="isbi*mprage_pp.")
        listaFLAIR_3=keyword_toList(path=lib_path,keyword="isbi*flair")
        listaSEG1_3=keyword_toList(path=lib_path,keyword="isbi*mask1")
        listaSEG2_3=keyword_toList(path=lib_path,keyword="isbi*mask2")

        update_labeled_folder(listaT1_2,listaFLAIR_2,listaSEG_2,listaMASK=None,datafolder=datafolder_val,nbNN=nbNN,ps=ps,numbernotnullpatch=10)
        update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)
        update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)
        update_labeled_folder(listaT1_1,listaFLAIR_1,listaSEG_1,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=30)

    t0=time.time()


    # load model (UNET3D)
    drop=0.5
    nf = 24
    if(modality=='T1_FLAIR'):
        in_dim=2
    else:
        in_dim=1
    model = modelos.unet_assemblynet(nf,2,drop,in_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion= lambda x,y : losses.mdice_loss_pytorch(x,y)

    if(regularized):
        filepath='weights/SUPERVISED_regularized.h5'
    else:
        filepath='weights/SUPERVISED_noreg.h5'

    

    #numb_data= len(sorted(glob.glob(datafolder+"x*.npy")))
    #numb_data_val= len(sorted(glob.glob(datafolder_val+"x*.npy")))

    transform_list=[]

    transform_list.append(ToTensor())


    data_transform = transforms.Compose(transform_list)
    dataset_train = TileDataset(datafolder,transform=data_transform,da=da_type)
    dataset_val = TileDataset(datafolder_val,transform=data_transform,da=False)
    dataset_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    dataset_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    best_val_loss= train_model(model=model,optimizer=optimizer,criterion=criterion,Epoch=Epoch,regularized=True,  loss_weights=[1,0.01],
            dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=criterion,eval_strategy='classic',
            out_PATH=filepath,early_stop=True,early_stop_treshold=early_stop_treshold)


    if (Snapshot):
        model= torch.load(filepath) # To get the best one
        for j in range(0,early_stop_treshold):
            print("Run=",j)
            best_val_loss= train_model(model=model,optimizer=optimizer,criterion=criterion,Epoch=1,regularized=True,  loss_weights=[1,0.01],
                        dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=criterion,eval_strategy='classic',
                        out_PATH=filepath, best_val_loss=best_val_loss)

            #model ensemble regularization (snapshot)
            if(j==0):
                moving_average_model = copy.deepcopy(model)
            else:
                #changing_ratio=1/(j+1)
                changing_ratio= 0.1
                moving_average_weights(running_model=model,stable_model=moving_average_model,changing_ratio=changing_ratio )
                moving_average_model = copy.deepcopy(model)


                    


