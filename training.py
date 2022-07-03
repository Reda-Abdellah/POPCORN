import os, glob, sys
import numpy as np
import nibabel as nii
import modelos
from utils import *
from losses import *
import torch
from helper import *
import torch.optim as optim
import segmentation_models_pytorch as smp

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
    criterion= lambda x,y : mdice_loss_pytorch(x,y)

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
    criterion= lambda x,y : mdice_loss_pytorch(x,y)

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


def train_on_labeled_only_Flare(regularized=True,save=True,Epoch=100, datafolder='dataset/train3D/',datafolder_val='dataset/val3D/', loss_weights=[1,0.01],batch_size=1 ,early_stop_treshold=30, da_type="iqda_v2", nbNN=[5,5,5],ps=[64,64,64],Snapshot=True, k_list=[0]): 

    for k in k_list:
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

            lib_path = os.path.join("..","flare_22_dataset/")
            
            listaT1=keyword_toList(path=lib_path,keyword="labelled_data/*.gz")
            listaSEG=keyword_toList(path=lib_path,keyword="labels/*.gz")
            listaT1= np.array(listaT1)
            listaSEG= np.array(listaSEG)
            val_ratio=0.8

            threshold=int(len(listaT1)*val_ratio)
            ind=np.arange(len(listaT1))
            np.random.seed(k)
            np.random.shuffle(ind)
            ind_train=ind[:threshold].tolist()
            ind_val=ind[threshold:].tolist()

            update_labeled_folder_flare(listaIM=listaT1[ind_train],listaSEG=listaSEG[ind_train], datafolder=datafolder, nbNN=nbNN,ps=ps,numbernotnullpatch=10, augment_times=6)
            update_labeled_folder_flare(listaIM=listaT1[ind_train],listaSEG=listaSEG[ind_train], datafolder=datafolder,nbNN=nbNN,ps=ps,numbernotnullpatch=27)
            update_labeled_folder_flare(listaIM=listaT1[ind_val],listaSEG=listaSEG[ind_val], datafolder=datafolder_val,nbNN=nbNN,ps=ps,numbernotnullpatch=27)


    t0=time.time()


    # load model (UNET3D)
    drop=0.5
    nf = 24
    in_dim=1
    model = modelos.unet_assemblynet(nf,13,drop,in_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion= lambda x,y : mdice_loss_pytorch(x,y)

    if(regularized):
        filepath='weights/SUPERVISED_flare_regularized.pt'
    else:
        filepath='weights/SUPERVISED_flare_noreg.h5'


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


def train_on_labeled_only2D(regularized=True,save=True,Epoch=100, datafolder='dataset/train/',datafolder_val='dataset/val/', loss_weights=[1,0.01],batch_size=1 ,early_stop_treshold=30, da_type="iqda_v2",modality='T1_FLAIR', img_size=[128,128],Snapshot=True, model_name="resnet18", loss_type="dice", k_list=[0], axis=2):
    
    for k in k_list:
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

            lib_path = os.path.join("..","flare_22_dataset/")
            
            listaT1=keyword_toList(path=lib_path,keyword="labelled_images/*.gz")
            listaSEG=keyword_toList(path=lib_path,keyword="labels/*.gz")
            listaT1= np.array(listaT1)
            listaSEG= np.array(listaSEG)
            val_ratio=0.8

            threshold=int(len(listaT1)*val_ratio)
            ind=np.arange(len(listaT1))
            np.random.seed(k)
            np.random.shuffle(ind)
            ind_train=ind[:threshold].tolist()
            ind_val=ind[threshold:].tolist()

            update_labeled_folder2D(listaIM=listaT1[ind_train],listaSEG=listaSEG[ind_train], datafolder=datafolder, augment_times=6, bg=20, step=2, all_slices=False,axis=axis)
            update_labeled_folder2D(listaIM=listaT1[ind_train],listaSEG=listaSEG[ind_train], datafolder=datafolder,bg=20, step=2,all_slices=True,axis=axis)
            update_labeled_folder2D(listaIM=listaT1[ind_val],listaSEG=listaSEG[ind_val], datafolder=datafolder_val,bg=20, step=2,all_slices=True,axis=axis)

        t0=time.time()


        # load model (UNET3D)
        drop=0.5
        nf = 24
        in_dim=1
        n_classes=13

        if(model_name=="resnet18"):
            model= smp.Unet( encoder_name="resnet18", activation='softmax', encoder_weights=None,  in_channels=in_dim, classes=n_classes)#, aux_params= {"classes" : n_classes, "pooling" : "max", #"avg". Default is "avg"  
        elif(model_name=="resnet50"):
            model= smp.Unet( encoder_name="resnet50", activation='softmax', encoder_weights=None,  in_channels=in_dim, classes=n_classes)#, aux_params= {"classes" : n_classes, "pooling" : "max", #"avg". Default is "avg"  
        #"dropout" :drop,  "activation": 'softmax'} )
            #pooling = "max",dropout =drop)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        
        if(loss_type=="dice"):
            criterion= lambda x,y : mdice_loss_pytorch2D(x,y)
        elif(loss_type=="DC_and_CE"):
            loss= DC_and_CE_loss_custom2D(1,1)
            criterion= lambda x,y : loss.loss(x,y)
        elif(loss_type=="CE"):
            criterion= lambda x,y : CE(x,y)

        val_criterion= lambda x,y : mdice_loss_pytorch2D(x,y)
        
        
        if(regularized):
            filepath='weights/SUPERVISED_2D_'+modality+'_regularized'+str(img_size[0])+'_'+da_type+model_name+loss_type+"_axis_"+str(axis)+"_k"+str(k)+'_.pt'
        else:
            filepath='weights/SUPERVISED_2D_'+modality+'_noreg'+str(img_size[0])+'_'+da_type+model_name+loss_type+"_axis_"+str(axis)+"_k"+str(k)+'.pt'


        if(regularized):
            dataset_train = TileDataset_with_reg_2D(datafolder,transform=None,da=da_type, img_size= img_size)
        else:
            dataset_train = TileDataset2D(datafolder,transform=None,da=da_type, img_size= img_size)
        dataset_val = TileDataset2D(datafolder_val,transform=None,da=False, img_size=img_size)
        dataset_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
        dataset_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        best_val_loss= train_model_2D(model=model,optimizer=optimizer,criterion=criterion,Epoch=Epoch,regularized=True,  loss_weights=loss_weights,
                dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=val_criterion,eval_strategy='classic',
                out_PATH=filepath,early_stop=True,early_stop_treshold=early_stop_treshold)


        if (Snapshot):
            model= torch.load(filepath) # To get the best one
            for j in range(0,early_stop_treshold):
                print("Run=",j)
                best_val_loss= train_model_2D(model=model,optimizer=optimizer,criterion=criterion,Epoch=1,regularized=True,  loss_weights=loss_weights,
                            dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=val_criterion,eval_strategy='classic',
                            out_PATH=filepath, best_val_loss=best_val_loss)

                #model ensemble regularization (snapshot)
                if(j==0):
                    moving_average_model = copy.deepcopy(model)
                else:
                    #changing_ratio=1/(j+1)
                    changing_ratio= 0.1
                    moving_average_weights(running_model=model,stable_model=moving_average_model,changing_ratio=changing_ratio )
                    moving_average_model = copy.deepcopy(model)


def POPCORN2D( img_size=[128,128], dataset_path="/lib/", Epoch_per_step=5, increment=20, datafolder='data_nearest2D/',datafolder_val='data_nearest_val2D/',dataselection_strategy='nearest',modality="T1_FLAIR",batch_size=4,
                resume=False, resume_after_adding_pseudo_of_step=1, load_precomputed_features=False, unlabeled_dataset="",recompute_distance_each_step=False,Snapshot=True, da_type="mixup",
                  regularized=True, loss_weights=[1,0.01], k=5, in_filepath="FLARE"):

    
    out_filepath= lambda x, y: 'weights/'+sys.argv[0].replace('.py','_step')+"%02d" % (x)+'_axis'+str(y)+'.pt'
    best_weight= lambda y:'weights/'+sys.argv[0].replace('.py','')+'_best_axis'+str(y)+'.pt'
    model=torch.load(in_filepath)
    
    lib_path = os.path.join("..","dataset/Subtask2")
    
    if(unlabeled_dataset=="FLARE"):
        listaT1= sorted(glob.glob(lib_path+"/unlabelled_images_part1/*.nii*"))+sorted(glob.glob(lib_path+"/unlabelled_images_part2/*.nii*"))
        listaT1= np.array(listaT1)
            
    listaT1_labeled=keyword_toList(path=lib_path,keyword="labelled_images/*.gz")
    listaSEG=keyword_toList(path=lib_path,keyword="labels/*.gz")
    listaT1= np.array(listaT1)
    listaSEG= np.array(listaSEG)
    
    if(not resume): 
        try:
            if(os.path.exists(datafolder+str(axis))):
                shutil.rmtree(datafolder+str(axis))
            os.mkdir(datafolder+str(axis))
        except OSError:
            print ("Creation of the directory %s failed" % datafolder+str(axis))
        else:
            print ("Successfully created the directory %s " % datafolder+str(axis))

        for axis in [0, 1, 2]:
            update_labeled_folder2D(listaIM=listaT1,listaSEG=listaSEG, datafolder=datafolder+str(axis), augment_times=2, bg=20, step=4, all_slices=False,axis=axis)
            update_labeled_folder2D(listaIM=listaT1,listaSEG=listaSEG, datafolder=datafolder+str(axis),bg=20, step=4,all_slices=True,axis=axis)
        
    #print(listaFLAIR_labeled)

    unlabeled_indxs= [i for i in range(len(listaT1))]
    pseudolabeled_indxs=[]
    unlabeled_num=len(unlabeled_indxs)
    pseudolabeled_num=len(pseudolabeled_indxs)
    labeled_num=len(listaT1_labeled)

    """
    transform_list=[]
    transform_list.append(ToTensor())
    data_transform = transforms.Compose(transform_list)
    dataset_train = TileDataset_with_reg(datafolder,transform=data_transform,da=da_type)
    """
    
    step=0
    Models= [torch.load(in_filepath[0]), torch.load(in_filepath[1]), torch.load(in_filepath[2])]
    Optimizers = [optim.Adam(Models[0].parameters(), lr=0.0001), optim.Adam(Models[1].parameters(), lr=0.0001), optim.Adam(Models[2].parameters(), lr=0.0001)] 
    criterion= lambda x,y : mdice_loss_pytorch2D(x,y)

    if( (not recompute_distance_each_step) and dataselection_strategy=='nearest'):
        if(load_precomputed_features or resume):
            print('loading precomputed features...')
            rank_distance=np.load('rank_distance_volbrain_tsne3_regularized2D.npy')
        else:
            print('computing bottleneck_features...')
            bottleneck_features_labeled,file_names= features_from_names_pytorch2D(listaT1_labeled, None, model)
            bottleneck_features_unlabeled,file_names= features_from_names_pytorch2D(listaT1, unlabeled_indxs ,model)
            print(bottleneck_features_unlabeled.shape)
            print('computing projection to simplar dataplane...')
            rank_distance= tsne_rank(bottleneck_features_unlabeled,bottleneck_features_labeled,n_components=3)
            np.save('rank_distance_volbrain_tsne3_regularized2D.npy',rank_distance)

    if( resume):
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
        if(not step<1):
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
            #update_data_folder_pytorch2D(Models,new_pos_in_features,listaT1,datafolder=datafolder)
        else:
            
            if(dataselection_strategy=='nearest'):
                if(recompute_distance_each_step):
                    bottleneck_features_labeled,file_names= features_from_names_pytorch2D(listaT1_labeled, None, model)
                    bottleneck_features_unlabeled,file_names= features_from_names_pytorch2D(listaT1,  unlabeled_indxs ,model, listaMASK)
                   
                    if(len(pseudolabeled_indxs)>0):
                        bottleneck_features_pseudolabeled,file_names= features_from_names_pytorch2D(listaT1, pseudolabeled_indxs ,model, listaMASK)
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
            update_data_folder_pytorch2D(Models,new_pos_in_features,listaT1,datafolder=datafolder)
        
        
        transform_list=[]
        transform_list.append(ToTensor())
        data_transform = transforms.Compose(transform_list)

        for axis in [0, 1, 2]:
            model= Models[axis]
            optimizer= Optimizers[axis]

            if(regularized):
                dataset_train = TileDataset_with_reg_2D(datafolder+str(axis),transform=None,da=da_type, img_size= img_size)
            else:
                dataset_train = TileDataset2D(datafolder+str(axis),transform=None,da=da_type, img_size= img_size)
            #dataset_val = TileDataset2D(datafolder_val,transform=None,da=False, img_size=img_size)
            dataset_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
            #dataset_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

            best_val_loss= train_model_2D(model=model,optimizer=optimizer,criterion=criterion,Epoch=Epoch_per_step,regularized=True,  loss_weights=loss_weights,
                    dataset_loader=dataset_loader,dataset_loader_val=None,val_criterion=criterion,eval_strategy='classic',  out_PATH=best_weight(axis))
            torch.save(model, out_filepath(step, axis))
    
    if (Snapshot):
        for axis in [0, 1, 2]:
            model= torch.load(out_filepath(step, axis)) # To get the best one
            optimizer= optim.Adam(model.parameters(), lr=0.0001)
            if(regularized):
                dataset_train = TileDataset_with_reg_2D(datafolder+str(axis),transform=None,da=da_type, img_size= img_size)
            else:
                dataset_train = TileDataset2D(datafolder+str(axis),transform=None,da=da_type, img_size= img_size)
            #dataset_val = TileDataset2D(datafolder_val,transform=None,da=False, img_size=img_size)
            dataset_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
            
            for j in range(0,10):
                print("Run=",j)
                best_val_loss= train_model_2D(model=model,optimizer=optimizer,criterion=criterion,Epoch=1,regularized=True,  loss_weights=loss_weights,
                                dataset_loader=dataset_loader,dataset_loader_val=dataset_loader_val,val_criterion=val_criterion,eval_strategy='classic',
                                out_PATH=best_weight(axis), best_val_loss=best_val_loss)

                #model ensemble regularization (snapshot)
                if(j==0):
                    moving_average_model = copy.deepcopy(model)
                else:
                    #changing_ratio=1/(j+1)
                    changing_ratio= 0.1
                    moving_average_weights(running_model=model,stable_model=moving_average_model,changing_ratio=changing_ratio )
                    moving_average_model = copy.deepcopy(model)
        
            torch.save(model, out_filepath(99, axis))
