import warnings
warnings.filterwarnings("ignore")
import torch
import pickle
from config import Config
from NetworkGPT import NetworkGPT
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # 1. Setting parameters-Training settings
    traning = False  # Whether to train, if False, please fill in the pre-trained model path
    train_model_file = 'pre_train/simulation/Simu_TP_30_TN_5_epoch_200.pth'

    # 2. Setting parameters-Some other basic settings
    args = Config()  # Loading parameters
    args.num_epoch = 200
    train_filename = 'pathway/simulation/SERGIO_perturbed_30_5_node_stable.data'
    test_filename = 'pathway/simulation/SERGIO_perturbed_30_5_for_test_stable.data'
    args.pca_file = 'result/simu_database_pca_reg_model.pkl'
    args.save_label = "Simu_reg"
    args.test_pathway = None    # To build a network for some genes, please enter the KEGG ID or a list of table files. You can fill in None
    args.n_rep = 1
    args.n_job = 1
    args.net_key_par = {'Flag_reg': True, 'Flag_llm': False, 'LLM_metric': 'euclidean'}

    # 3. Training process
    trainer = NetworkGPT(args)
    if traning:
        best_mean_AUC, train_model_file, printf = trainer.train(train_filename, n_train=100, n_test=[1000, 1000+10])
        with open('Run_all_sample_on_simulation.txt', 'a') as file:
            file.write(train_model_file + '\n')
            file.write(str(printf) + '\n')

    # 4. Network generation and evaluation
    diffusion_pre = torch.load(train_model_file, map_location=trainer.device)
    results = {'AUC': [], 'AUPR': [], 'F1': []}
    result_filename = 'Simulation_reg_test_305.data'
    par = tqdm(range(0, 100))
    for test_num in par:
        testdata, truelabel = trainer.load_test_data(test_filename, num=test_num)
        adj_final = trainer.test(diffusion_pre, testdata)
        performance = trainer.evaluation(adj_final, truelabel)
        results['AUC'].append(performance['AUC'])
        results['AUPR'].append(performance['AUPR'])
        results['F1'].append(performance['F1'])
        f = open(result_filename, 'wb')
        pickle.dump(results, f)
        f.close()
        par.set_description(f"The: {test_num + 1:4.0f}-th  gene expression profile. AUROC: {performance['AUC']:.4f} ; AUPRC: {performance['AUPR']:.4f}")

    golbalmeanAUC = np.mean(results['AUC'])
    golbalmeanAUPR = np.mean(results['AUPR'])
    golbalmeanF1 = np.mean(results['F1'])
    print(f"NetworkGPT - AUROC average: {golbalmeanAUC:.4f} AUPRC average: {golbalmeanAUPR:.4f}  F1-score average: {golbalmeanF1:.4f}")
    print(f'**************   Task is finished! The result are saved in {str(result_filename)} !   **************')

