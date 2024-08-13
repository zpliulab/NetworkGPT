import warnings
warnings.filterwarnings("ignore")
from config import Config
from NetGPT import NetGPT
import os
import torch

if __name__ == '__main__':
    # 1. Setting parameters-Training settings
    traning = False  # Whether to train, if False, please fill in the pre-trained model path
    train_model_file = 'pre_train/HCC/Training_HCC_Tumor_Tcell_data.pth'
    tis = 'Tumor'    # tissue = ['Tumor', 'Adjacent_liver']
    cell = 'Tcell'   # celltype = ['Tcell', 'Bcell', 'NK', 'Mye','HSC', 'Tumor']

    # 2. Setting parameters-Some other basic settings
    args = Config()  # Loading parameters
    args.test_pathway = "hsa05225"  # HCC KEGG pathway
    args.num_epoch = 600
    args.net_key_par = {'Flag_reg': True, 'Flag_llm': True, 'LLM_metric': 'euclidean'}
    args.n_job = 1
    args.theta_PMI = 40              # PMI threshold
    args.theta_PPC = 40              # PPC threshold
    args.database = ['STRING', 'KEGG']  # Database name: RegNetwork, STRING, TRRUST, KEGG or None
    train_filename = 'Cancer_datasets/HCC/HCC_' + str(tis) + '_' + str(cell) + '_input.csv'
    test_filename = train_filename
    args.pca_file = 'result/HCC_' + str(tis) + '_' + str(cell) + '_pca_model.pkl'
    args.save_label = 'Training_HCC_' + str(tis) + '_' + str(cell) + '_data'
    args.LLM_filepath = 'LLM/Geneformer_' + str(tis) + '_' + str(
        cell) + '_HCC_output.csv'  # LLM path scFoundation, Geneformer, BioBERT, or None
    trainer = NetGPT(args)

    # 3. Training process
    if traning:
        best_mean_AUC, train_model_file, printf = trainer.train(train_filename)
        with open('Run_all_sample_HCC.txt', 'a') as file:
            file.write(train_model_file + '\n')
            file.write(str(printf) + '\n')

    # 4. Load pretrained model and test data
    diffusion_pre = torch.load(train_model_file, map_location=trainer.device)
    testdata, truelabel = trainer.load_test_data(test_filename)

    # 5. Network generation and evaluation
    adj_final = trainer.test(diffusion_pre, testdata)  # 训练模拟数据
    performance = trainer.evaluation(adj_final, truelabel)

    print(f"HCC Gene expression profile {tis:s}-{cell:s}. AUROC: {performance['AUC']:.4f} ; AUPRC: {performance['AUPR']:.4f}")

