import os
import requests

folder_name = "GRN"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


file_path = "GRN/TF.txt"
if os.path.exists(file_path):
    print("TF.txt 文件存在")
else:
    url = "https://guolab.wchscu.cn/static/AnimalTFDB3/download/Homo_sapiens_TF"
    file_path = os.path.join(folder_name, "TF.txt")
    response = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(response.content)

    print("TF文件下载完成。")

file_path = "GRN/Genome.txt"
if os.path.exists(file_path):
    print("Genome.txt 文件存在")
else:
    url = "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt" # HGNC数据库
    file_path = os.path.join(folder_name, "Genome.txt")
    response = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(response.content)
    print("Genome文件下载完成。")








