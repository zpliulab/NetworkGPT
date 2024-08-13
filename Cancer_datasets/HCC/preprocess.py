import pandas as pd
import numpy as np
import subprocess

HCC_exp_file = 'HCC_log_tpm_expression_matrix.txt'
HCC_exp = pd.read_csv(HCC_exp_file, sep='\t')
HCC_exp.set_index(HCC_exp.columns[0], inplace=True)
HCC_exp.drop(columns=HCC_exp.columns[0], inplace=True)

metadata_file = 'HCC_cell_metadata.txt'
metadata = pd.read_csv(metadata_file, sep='\t', header=[0, 1])

metadata.columns = ['_'.join(col).strip() for col in metadata.columns.values]
metadata[['group', 'cell_type']] = metadata['cell_type_group'].str.split('_', expand=True)
metadata['cell_type'] = metadata['cell_type'].str.replace('.', '', regex=False)
metadata['tissue_source_group'] = metadata['tissue_source_group'].str.replace(' ', '_', regex=False)

cell_type = pd.unique(metadata['cell_type'])
disease_type = pd.unique(metadata['tissue_source_group'])
for celli in cell_type:
    for dis in disease_type:
        sub_metadata = metadata[metadata['cell_type'] == celli]
        sub_metadata = sub_metadata[sub_metadata['tissue_source_group'] == dis]['sample_name_sample_attribute']
        sub_metadata = sub_metadata[sub_metadata.isin(HCC_exp.columns)]
        sub_HCC_exp = HCC_exp[sub_metadata]

        row_zero_ratio = (sub_HCC_exp == 0).sum(axis=1) / sub_HCC_exp.shape[1]
        col_zero_ratio = (sub_HCC_exp == 0).sum(axis=0) / sub_HCC_exp.shape[0]
        sub_HCC_exp_filter = sub_HCC_exp[row_zero_ratio < 0.95]
        sub_HCC_exp_filter = sub_HCC_exp_filter.loc[:, col_zero_ratio < 0.95]

        sub_HCC_exp_filter.to_csv('./preprocess/HCC_'+dis+'_'+celli.replace('.', '')+'_input.csv', index=True)
        shape1 = sub_HCC_exp_filter.shape
        print(f'HCC_{dis}_{celli} finished!')