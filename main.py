# from tqdm import tqdm
# from typing import Optional, List, Tuple
# import torch 
import os
from docstalks.utils import (read_pdf_in_document,
                             split_text_by_chunks,
                             add_embeddings_to_document,
                             )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


data_path = "/Users/eugene/Desktop/SoftTeco/danswer/data-softteco"

fnames = os.listdir(data_path)
flist = [os.path.join(data_path, fname) for fname in fnames]

print(f"INFO: Files in data storage: {len(flist)}")