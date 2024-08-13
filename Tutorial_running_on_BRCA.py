import warnings
warnings.filterwarnings("ignore")
from config import Config
from NetworkGPT import NetworkGPT
import torch

if __name__ == '__main__':
    # 1. Setting parameters-Training settings
    traning = False  # Whether to train, if False, please fill in the pre-trained model path
    train_model_file = 'pre_train/BRCA/Training_reg_BRCA_Tumor_T_cell_data.pth'
    cell = 'T_cell'  # celltype = ['T_cell', 'B_cell', 'Myeloid', 'Cancer', 'DC', 'EC', 'Fibroblast', 'Mast']

    # 2. Setting parameters-Some other basic settings
    args = Config()  # 加载参数
    args.test_pathway = "hsa05224"
    args.num_epoch = 600
    args.net_key_par = {'Flag_reg': True, 'Flag_llm': True, 'LLM_metric': 'euclidean'}
    args.n_job = 10
    args.theta_PMI = 40      # PMI threshold
    args.theta_PPC = 40      # PPC threshold
    args.database = ['STRING', 'KEGG']   # Database name: RegNetwork, STRING, KEGG, TRRUST  or None
    args.LLM_filepath = 'LLM/Geneformer_'+str(cell)+'_BRCA_output.csv'  # LLM path scFoundation, Geneformer, BioBERT, or None
    train_filename = 'Cancer_datasets/BRCA/BRCA_Tumor_'+str(cell)+'_output.csv'
    test_filename = train_filename
    args.pca_file = 'result/BRCA_Tumor_'+str(cell)+'_pca_model.pkl'
    args.save_label = 'Training_reg_BRCA_Tumor_'+str(cell)+'_data'
    trainer = NetworkGPT(args)

    # 3. Training process
    if traning:
        best_mean_AUC, train_model_file, printf = trainer.train(train_filename)
        with open('Run_all_sample_BRCA.txt', 'a') as file:
            file.write(train_model_file + '\n')
            file.write(str(printf) + '\n')

    # 4. Load pretrained model and test data
    diffusion_pre = torch.load(train_model_file, map_location=trainer.device)
    testdata, truelabel = trainer.load_test_data(test_filename)

    # 5. Network generation and evaluation
    adj_final = trainer.test(diffusion_pre, testdata)  # 训练模拟数据
    performance = trainer.evaluation(adj_final, truelabel)

    print( f"HCC Gene expression profile {cell:s}. AUROC: {performance['AUC']:.4f} ; AUPRC: {performance['AUPR']:.4f}")