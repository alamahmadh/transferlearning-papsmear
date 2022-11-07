import os

class Config():
    seed = 42
    num_workers = 0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_root = "samples"
    model_path = "models/checkpoints"
    #train_csv = os.path.join(data_root, 'csv_train.csv')
    #val_csv = os.path.join(data_root, 'csv_val.csv')
    #test_csv = os.path.join(data_root, 'csv_test.csv')
    
    model = 'resnet18'
    
    batch_size = 4
    num_epochs = 4
    lr = 0.001
    # Flag for feature extracting. When False, we finetune the whole model,
    # When True we only update the reshaped layer params
    feature_extract = True
    
    optimizer = 'adam'