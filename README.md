# GAHs
Guided Attention Heads


## steps
1. Put your datset in the folder datasets/your_DataSet_name/your_data_file, then update the load in preprocessor/raw_data_loader.py

2. Preprocess your data (semantic annotation): python main.py run_mode preprocess --dataset TREC --splits train,test

-- the processed data will be in datasets/your_DataSet_name/**
-- splits means how man files you have

3. Train the model: train.py


