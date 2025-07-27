# -*- encoding: utf-8 -*-

import random
from typing import List
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import XLNetModel, XLNetTokenizer
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import OneCycleLR

EPOCHS = 1
BATCH_SIZE = 16
LR = 1e-4 
MAXLEN = 64
POOLING = 'cls'   
DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu') 

# Pre-trained model directory
XLNET = 'xlnet-base-cased'  
model_path = XLNET

# Location for storing fine-tuned parameters
SAVE_PATH = './saved_model_misogyny/simcse_sup_xlnet.pt'

# 数据位置
TRAIN = './simcse-datasets/misogyny/train.txt'
DEV = './simcse-datasets/misogyny/dev.txt'
TEST = './simcse-datasets/misogyny/test.txt'

#

def load_data(train_or_not,path: str) -> List:
    """
    """
    #TODO: 
    if train_or_not:  
        with jsonlines.open(path, 'r') as f:
            
            return [(line['misogyny1'], line['misogyny2'], line['non-misogyny']) for line in f]
    else:
        with jsonlines.open(path, 'r') as f:
            return [(line['source'], line['content'], line['label']) for line in f]


class TrainDataset(Dataset):
    
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1], text[2]], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    
    
class TestDataset(Dataset):
    
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, 
                         padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2])
    

        
class ClaDModel(nn.Module):
    
    def __init__(self, pretrained_model: str, pooling: str, freeze_layers: int = 6):
        super(ClaDModel, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(pretrained_model)
        self.pooling = pooling

        #Add additional neural network layers
        self.fc1 = nn.Linear(768, 512)  #  XLNet output is 768-dimensional.
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)    

        # Freeze a fixed number of layers
        self.freeze_layers(freeze_layers)

    def freeze_layers(self, freeze_layers: int):
        for i, param in enumerate(self.xlnet.layer):
            if i < freeze_layers:
                for param in param.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        if self.pooling == 'cls':
            pooled_output = outputs.last_hidden_state[:, 0]  # [batch, 768]
        elif self.pooling == 'last-avg':
            last = outputs.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            pooled_output = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif self.pooling == 'first-last-avg':
            first = outputs.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = outputs.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            pooled_output = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        
        x = F.relu(self.fc1(pooled_output))
        x = F.relu(self.fc2(x))
        binary_output = torch.sigmoid(self.fc3(x))  

        return pooled_output, binary_output   


                  
# Initialize the sliding window
window_size = 100  # Define the sliding window size
window_data = []

def update_window(window_data, new_data, window_size):
    # Create a mask to filter the 3rd, 6th, 9th, ... data in each batch.
    mask = torch.ones(new_data.size(0), dtype=torch.bool)
    mask[2::3] = False  # Starting from index 2, set every 3rd element to False.
    filtered_data = new_data[mask]
    window_data.append(filtered_data)
    if len(window_data) > window_size:
        window_data.pop(0)
    return window_data

def compute_window_statistics(window_data):
    window_data_concatenated = torch.cat(window_data, dim=0)
    window_data_concatenated_unique = torch.unique(window_data_concatenated, dim=0)
    mean = torch.mean(window_data_concatenated_unique, dim=0, keepdim=True)
    std = torch.std(window_data_concatenated_unique, dim=0, unbiased=False, keepdim=True)
    normalized_data = (window_data_concatenated_unique - mean) / (std + 1e-7)
    covariance_matrix = torch.cov(normalized_data.T)

    return mean, covariance_matrix,std




def simcse_sup_loss_mahalanobis_update(y_pred, mean, std, covariance_matrix,class_weights):
    batch_size = y_pred.shape[0] // 3
    normalized_y_pred = (y_pred - mean) / (std + 1e-7)

    d = covariance_matrix.size(0)
    regularization_term = 1e-4
    identity_matrix = torch.eye(d, device=covariance_matrix.device) * regularization_term
    regularized_covariance_matrix = covariance_matrix + identity_matrix
    inv_covariance_matrix = torch.inverse(regularized_covariance_matrix)

    diff = normalized_y_pred.squeeze(0)
    mahalanobis_distance_sq_tensor = torch.tensor([], device=y_pred.device)
    
    for i in range(diff.size(0)):
        diff_i = diff[i].view(1, -1)
        mahalanobis_distance_sq = torch.mm(torch.mm(diff_i, inv_covariance_matrix), diff_i.t())
        mahalanobis_distance_sq_tensor = torch.cat((mahalanobis_distance_sq_tensor, mahalanobis_distance_sq.view(-1)), dim=0)

    similar_distances = mahalanobis_distance_sq_tensor.view(batch_size, 3)[:, :2].flatten()
    dissimilar_distances = mahalanobis_distance_sq_tensor.view(batch_size, 3)[:, 2]

    similarity_scores = torch.exp(-similar_distances/1000)
    dissimilarity_scores = torch.exp(-dissimilar_distances/1000)

    weighted_similarity_scores = similarity_scores 
    weighted_dissimilarity_scores = dissimilarity_scores 

    combined_scores = torch.cat([weighted_similarity_scores, weighted_dissimilarity_scores], dim=0).view(-1, 1)
    combined_scores = torch.cat([combined_scores, 1 - combined_scores], dim=1)

    y_true_combined = torch.cat([torch.zeros(batch_size * 2, dtype=torch.long, device=y_pred.device),
                                 torch.ones(batch_size, dtype=torch.long, device=y_pred.device)], dim=0)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=y_pred.device)

    loss = F.cross_entropy(combined_scores, y_true_combined,weight=class_weights)
    return loss




def eval_mal(model, dataloader, mean, covariance_matrix, std) -> float:
    """Model evaluation function using Mahalanobis distance"""
    model.eval()
    label_array = np.array([])
    inv_covariance_matrix = torch.inverse(covariance_matrix)
    mahalanobis_distance_sq_tensor = torch.tensor([], device=DEVICE)

    with torch.no_grad():
        for source, target, label in dataloader:
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            pooled_output_source, _ = model(source_input_ids, source_attention_mask, source_token_type_ids)

            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
            pooled_output_target, _ = model(target_input_ids, target_attention_mask, target_token_type_ids)
            
            target_pred_normalized = (pooled_output_target - mean) / (std + 1e-7)
            
            diff = target_pred_normalized.squeeze(0)
            
            if diff.dim() == 1:
                diff = diff.view(1, -1)  
          
            for i in range(diff.size(0)):
                diff_i = diff[i].view(1, -1)
                
                mahalanobis_distance_sq = torch.mm(torch.mm(diff_i, inv_covariance_matrix), diff_i.t())
                mahalanobis_distance_sq_tensor = torch.cat((mahalanobis_distance_sq_tensor, mahalanobis_distance_sq.view(-1)), dim=0)

            label_array = np.append(label_array, np.array(label))

    
    normalized_similarity = 1.0 / (1.0 + mahalanobis_distance_sq_tensor)
    
    return spearmanr(label_array, normalized_similarity.cpu().numpy()).correlation


def eval_binary(model, dataloader) -> float:
    
    model.eval()
    label_array = np.array([])
    correct_predictions = 0  
    total_predictions = 0    

    with torch.no_grad():
        for source, target, label in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            
            label_array = np.append(label_array, np.array(label))
            
            _, binary_output = model(source_input_ids, source_attention_mask, source_token_type_ids)

            
            predictions = binary_output.round().squeeze()  
            correct_predictions += (predictions == label.to(DEVICE)).sum().item()
            total_predictions += label.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy



def print_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print('Gradient norm: ', total_norm)


def train(model, train_dl, dev_dl, optimizer, scheduler) -> None:
    """Model training function"""
    model.train()
    
    global best, window_data, mean, covariance_matrix, std
    early_stop_batch = 0
    best = -float('inf')
    
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)

        pooled_output, binary_output = model(input_ids, attention_mask, token_type_ids)

        # Calculate the loss and perform backpropagation
        out_detached = pooled_output.detach()
        window_data = update_window(window_data, out_detached, window_size)
        mean, covariance_matrix, std = compute_window_statistics(window_data)
        loss = simcse_sup_loss_mahalanobis_update(pooled_output, mean, std, covariance_matrix, [100, 1000])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # Update scheduler
        scheduler.step()  
        # Eval
        if batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']  
            logger.info(f'Current learning rate: {current_lr:.6f}')  
            logger.info(f'loss: {loss.item():.4f}')
            corrcoef = eval_mal(model, dev_dl, mean, covariance_matrix, std)
            acc = eval_binary(model, dev_dl)
            print(f"corrcoef: {corrcoef:.4f} in batch: {batch_idx}")
            print(f"acc: {acc:.4f} in batch: {batch_idx}")
            
            model.train()
            if best < corrcoef:
                best = corrcoef
                early_stop_batch = 0
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
            else:
                early_stop_batch += 1
                if early_stop_batch == 10:
                    logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
                    logger.info(f"train use sample number: {(batch_idx - 10) * BATCH_SIZE}")
                    return

    torch.save(model.state_dict(), SAVE_PATH)

if __name__ == '__main__':
    
    logger.info(f'device: {DEVICE}, pooling: {POOLING}')
    
    tokenizer = XLNetTokenizer.from_pretrained(model_path)
    
    # load data
    train_data = load_data(True, TRAIN)
    print(len(train_data))
    random.shuffle(train_data)                        
    dev_data = load_data(False, DEV)
    test_data = load_data(False, TEST)
    
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset(test_data), batch_size=BATCH_SIZE)
    
    # load model
    model = ClaDModel(pretrained_model=model_path, pooling=POOLING, freeze_layers=2)

    model.to(DEVICE)
    
    # Initialize optimizer and OneCycleLR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = OneCycleLR(optimizer, max_lr=5e-5, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)

    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer, scheduler)

    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    
    model.load_state_dict(torch.load(SAVE_PATH))
    dev_corrcoef = eval_mal(model, dev_dataloader, mean, covariance_matrix, std)
    test_corrcoef = eval_mal(model, test_dataloader, mean, covariance_matrix, std)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
    
    
    