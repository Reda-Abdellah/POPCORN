import os, glob, sys
import numpy as np
import nibabel as nii
import modelos
from utils import *
import losses
import torch
from helper import *
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)

def POPCORN( ps=[64,64,64], nbNN=[5,5,5], dataset_path="/lib/", Epoch_per_step=2, increment=200, datafolder='data_nearest/',datafolder_val='data_nearest_val/',dataselection_strategy='nearest',modality="T1_FLAIR",batch_size=4,
                resume=False, resume_after_adding_pseudo_of_step=1, load_precomputed_features=False, unlabeled_dataset="OFSEP_and_volbrain",recompute_distance_each_step=True,Snapshot=True, da_type="iqda_v2",
                  regularized=True, loss_weights=[1,0.01], k=5):

    
    if(regularized):
        in_filepath='weights/SUPERVISED_'+modality+'_regularized.pt'
    else:
        in_filepath='weights/SUPERVISED_'+modality+'_noreg.h5'
    
    out_filepath= lambda x: 'weights/'+sys.argv[0].replace('.py','_step')+"%02d" % (x)+'.pt'
    best_weight= 'weights/'+sys.argv[0].replace('.py','')+'_best.pt'
    model=torch.load(in_filepath)
    
    if(unlabeled_dataset=="volbrain"):
        listaT1= sorted(glob.glob(dataset_path+"/volbrain/n_mfmni*t1*.nii*"))
        listaFLAIR = sorted(glob.glob(dataset_path+"/volbrain/n_mfmni*flair*.nii*"))
        listaMASK = sorted(glob.glob(dataset_path+"/volbrain/mask*.nii*"))
        listaMASK = np.array(listaMASK)
        listaT1= np.array(listaT1)
        listaFLAIR= np.array(listaFLAIR)
    elif(unlabeled_dataset=="OFSEP_and_volbrain"):
        listaT1 = sorted(glob.glob(dataset_path+"/volbrain/n_mfmni*t1*.nii*"))+sorted(glob.glob(dataset_path+"/OFSEP/n_mfmni*T1*.nii*"))
        listaFLAIR = sorted(glob.glob(dataset_path+"/volbrain/n_mfmni*flair*.nii*"))+sorted(glob.glob(dataset_path+"/OFSEP/n_mfmni*FLAIR*.nii*"))
        listaMASK = sorted(glob.glob(dataset_path+"/volbrain/mask*.nii*"))+sorted(glob.glob(dataset_path+"/OFSEP/mask*.nii*"))
        listaMASK = np.array(listaMASK)#[:10]
        listaT1= np.array(listaT1)#[:10]
        listaFLAIR= np.array(listaFLAIR)#[:10]
   
    if(modality=='FLAIR'):
        listaT1=None

    lib_path = os.path.join("..","all_ms_preprocessed")
        
    listaT1_1=keyword_toList(path=lib_path,keyword="mso*mprage.")
    listaFLAIR_1=keyword_toList(path=lib_path,keyword="mso*flair")
    listaT1_2=keyword_toList(path=lib_path,keyword="msseg*mprage.")
    listaFLAIR_2=keyword_toList(path=lib_path,keyword="msseg*flair")
    listaSEG_2=keyword_toList(path=lib_path,keyword="msseg*mask1")
    listaSEG_1=keyword_toList(path=lib_path,keyword="mso*mask1")
    listaT1_3=keyword_toList(path=lib_path,keyword="isbi*mprage_pp.")
    listaFLAIR_3=keyword_toList(path=lib_path,keyword="isbi*flair")
    listaSEG1_3=keyword_toList(path=lib_path,keyword="isbi*mask1")
    listaSEG2_3=keyword_toList(path=lib_path,keyword="isbi*mask2")

    listaFLAIR_labeled= np.array(listaFLAIR_1+listaFLAIR_3)#[:10]
    listaT1_labeled= np.array(listaT1_1+listaT1_3)#[:10]
    if(not resume): 
        try:
            if(os.path.exists(datafolder)):
                shutil.rmtree(datafolder)
                os.mkdir(datafolder)
        except OSError:
            print ("Creation of the directory %s failed" % datafolder)
        else:
            print ("Successfully created the directory %s " % datafolder)

        try:
            if(os.path.exists(datafolder_val)):
                shutil.rmtree(datafolder_val)
            os.mkdir(datafolder_val)
        except OSError:
            print ("Creation of the directory %s failed" % datafolder_val)
        else:
            print ("Successfully created the directory %s " % datafolder_val)

        

        if(modality=='T1_FLAIR'):
            update_labeled_folder(listaT1_2,listaFLAIR_2,listaSEG_2,listaMASK=None,datafolder=datafolder_val,nbNN=nbNN,ps=ps,numbernotnullpatch=10)
            update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder(listaT1_1,listaFLAIR_1,listaSEG_1,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)
        else:
            update_labeled_folder_flair(listaT1_2,listaFLAIR_2,listaSEG_2,listaMASK=None,datafolder=datafolder_val,nbNN=nbNN,ps=ps,numbernotnullpatch=10)
            update_labeled_folder_flair(listaT1_3,listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder_flair(listaT1_3,listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder_flair(listaT1_1,listaFLAIR_1,listaSEG_1,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)
    
    #print(listaFLAIR_labeled)

    unlabeled_indxs= [i for i in range(len(listaFLAIR))]
    pseudolabeled_indxs=[]
    unlabeled_num=len(unlabeled_indxs)
    pseudolabeled_num=len(pseudolabeled_indxs)
    labeled_num=len(listaFLAIR_labeled)

    """
    transform_list=[]
    transform_list.append(ToTensor())
    data_transform = transforms.Compose(transform_list)
    dataset_train = TileDataset_with_reg(datafolder,transform=data_transform,da=da_type)
    """
    
    step=0
    model= torch.load(in_filepath)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion= lambda x,y : losses.mdice_loss_pytorch(x,y)

    if( (not recompute_distance_each_step) and dataselection_strategy=='nearest'):
        if(load_precomputed_features or resume):
            print('loading precomputed features...')
            rank_distance=np.load('rank_distance_volbrain_tsne3_regularized.npy')
        else:
            print('computing bottleneck_features...')
            bottleneck_features_labeled,file_names= features_from_names_pytorch(listaT1_labeled, listaFLAIR_labeled, None, model)
            bottleneck_features_unlabeled,file_names= features_from_names_pytorch(listaT1, listaFLAIR, unlabeled_indxs ,model, listaMASK)
            print(bottleneck_features_unlabeled.shape)
            print('computing projection to simplar dataplane...')
            rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
            np.save('rank_distance_volbrain_tsne3_regularized.npy',rank_distance)

    if(resume):
        for it in range(resume_after_adding_pseudo_of_step):
            increment_new_data= int(increment+ it* increment/2)
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
            model= torch.load(out_filepath(step))


    #Training
    increment_new_data= int(increment+ step* increment/2)
    while(unlabeled_num>increment_new_data):
        increment_new_data= int(increment+ step* increment/2)
        step=step+1
        
        print('step: '+str(step))
        print('loading new data...')
        if( resume and step==resume_after_adding_pseudo_of_step):
            print('resuming..')
            #update_data_folder_pytorch(model,new_pos_in_features[430:],listaT1,listaFLAIR,listaMASK,datafolder=datafolder)
        else:
            
            if(dataselection_strategy=='nearest'):
                if(recompute_distance_each_step):
                    bottleneck_features_labeled,file_names= features_from_names_pytorch(listaT1_labeled, listaFLAIR_labeled, None, model)
                    bottleneck_features_unlabeled,file_names= features_from_names_pytorch(listaT1, listaFLAIR, unlabeled_indxs ,model, listaMASK)
                   
                    if(len(pseudolabeled_indxs)>0):
                        bottleneck_features_pseudolabeled,file_names= features_from_names_pytorch(listaT1, listaFLAIR, pseudolabeled_indxs ,model, listaMASK)
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
            update_data_folder_pytorch(model,new_pos_in_features,listaT1,listaFLAIR,listaMASK,datafolder=datafolder)
        
        
        transform_list=[]
        transform_list.append(ToTensor())
        data_transform = transforms.Compose(transform_list)

        if(regularized):
            dataset_train = TileDataset_with_reg(datafolder,transform=data_transform,da=da_type)
        else:
            dataset_train = TileDataset(datafolder,transform=data_transform,da=da_type)
        dataset_val = TileDataset(datafolder_val,transform=data_transform,da=False)
        dataset_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
        dataset_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        best_val_loss= train_model(model=model,optimizer=optimizer,criterion=criterion,Epoch=Epoch_per_step,regularized=True,  loss_weights=loss_weights,
                dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=criterion,eval_strategy='classic',
                out_PATH=best_weight)

        torch.save(model, out_filepath(step))
    
    if (Snapshot):
        model= torch.load(filepath) # To get the best one
        for j in range(0,10):
            print("Run=",j)
            best_val_loss= train_model(model=model,optimizer=optimizer,criterion=criterion,Epoch=1,regularized=True,  loss_weights=loss_weights,
                        dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=criterion,eval_strategy='classic',
                        out_PATH=best_weight, best_val_loss=best_val_loss)

            #model ensemble regularization (snapshot)
            if(j==0):
                moving_average_model = copy.deepcopy(model)
            else:
                #changing_ratio=1/(j+1)
                changing_ratio= 0.1
                moving_average_weights(running_model=model,stable_model=moving_average_model,changing_ratio=changing_ratio )
                moving_average_model = copy.deepcopy(model)
    
    torch.save(model, out_filepath("snapshot"))

    


def train_on_labeled_only(regularized=True,save=True,Epoch=100, datafolder='dataset/train/',datafolder_val='dataset/val/', loss_weights=[1,0.01],batch_size=1 ,early_stop_treshold=30, da_type="iqda_v2",modality='T1_FLAIR', nbNN=[5,5,5],ps=[64,64,64],Snapshot=True):
    

    if(save):
        
        try:
            if(os.path.exists(datafolder)):
                shutil.rmtree(datafolder)
                os.mkdir(datafolder)
        except OSError:
            print ("Creation of the directory %s failed" % datafolder)
        else:
            print ("Successfully created the directory %s " % datafolder)

        try:
            if(os.path.exists(datafolder_val)):
                shutil.rmtree(datafolder_val)
            os.mkdir(datafolder_val)
        except OSError:
            print ("Creation of the directory %s failed" % datafolder_val)
        else:
            print ("Successfully created the directory %s " % datafolder_val)

        lib_path = os.path.join("..","all_ms_preprocessed")
        
        listaT1_1=keyword_toList(path=lib_path,keyword="mso*mprage.")
        listaFLAIR_1=keyword_toList(path=lib_path,keyword="mso*flair")
        listaT1_2=keyword_toList(path=lib_path,keyword="msseg*mprage.")
        listaFLAIR_2=keyword_toList(path=lib_path,keyword="msseg*flair")
        listaSEG_2=keyword_toList(path=lib_path,keyword="msseg*mask1")
        listaSEG_1=keyword_toList(path=lib_path,keyword="mso*mask1")
        listaT1_3=keyword_toList(path=lib_path,keyword="isbi*mprage_pp.")
        listaFLAIR_3=keyword_toList(path=lib_path,keyword="isbi*flair")
        listaSEG1_3=keyword_toList(path=lib_path,keyword="isbi*mask1")
        listaSEG2_3=keyword_toList(path=lib_path,keyword="isbi*mask2")

        if(modality=='T1_FLAIR'):
            update_labeled_folder(listaT1_2,listaFLAIR_2,listaSEG_2,listaMASK=None,datafolder=datafolder_val,nbNN=nbNN,ps=ps,numbernotnullpatch=10)
            update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder(listaT1_3,listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder(listaT1_1,listaFLAIR_1,listaSEG_1,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)
        else:
            update_labeled_folder_flair(listaT1_2,listaFLAIR_2,listaSEG_2,listaMASK=None,datafolder=datafolder_val,nbNN=nbNN,ps=ps,numbernotnullpatch=10)
            update_labeled_folder_flair(listaT1_3,listaFLAIR_3,listaSEG1_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder_flair(listaT1_3,listaFLAIR_3,listaSEG2_3,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=5)
            update_labeled_folder_flair(listaT1_1,listaFLAIR_1,listaSEG_1,listaMASK=None,datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=15)

    t0=time.time()


    # load model (UNET3D)
    drop=0.5
    nf = 24
    if(modality=='T1_FLAIR'):
        in_dim=2
    else:
        in_dim=1
    model = modelos.unet_assemblynet(nf,2,drop,in_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion= lambda x,y : losses.mdice_loss_pytorch(x,y)

    if(regularized):
        filepath='weights/SUPERVISED_'+modality+'_regularized.pt'
    else:
        filepath='weights/SUPERVISED_'+modality+'_noreg.h5'


    transform_list=[]

    transform_list.append(ToTensor())


    data_transform = transforms.Compose(transform_list)
    if(regularized):
        dataset_train = TileDataset_with_reg(datafolder,transform=data_transform,da=da_type)
    else:
        dataset_train = TileDataset(datafolder,transform=data_transform,da=da_type)
    dataset_val = TileDataset(datafolder_val,transform=data_transform,da=False)
    dataset_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    dataset_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    best_val_loss= train_model(model=model,optimizer=optimizer,criterion=criterion,Epoch=Epoch,regularized=True,  loss_weights=loss_weights,
            dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=criterion,eval_strategy='classic',
            out_PATH=filepath,early_stop=True,early_stop_treshold=early_stop_treshold)


    if (Snapshot):
        model= torch.load(filepath) # To get the best one
        for j in range(0,early_stop_treshold):
            print("Run=",j)
            best_val_loss= train_model(model=model,optimizer=optimizer,criterion=criterion,Epoch=1,regularized=True,  loss_weights=loss_weights,
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


                    


