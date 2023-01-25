from torch.utils.data import Dataset
from ts.exceptions import CustomException
import sys
import re

class NewsDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, source_len, target_len):
        try:
            self.source_texts = source_texts
            self.target_texts = target_texts
            self.tokenizer = tokenizer
            self.source_len = source_len
            self.target_len = target_len
        except Exception as e:
            raise CustomException(e, sys)
    
    def __len__(self):
        return len(self.target_texts) - 1
    
    def __getitem__(self, idx):
        whitespace_handler = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
        text = " ".join(str(self.source_texts[idx]).split())
        summary = " ".join(str(self.target_texts[idx]).split())
        
        source = self.tokenizer.batch_encode_plus([whitespace_handler(text)],
                                                max_length= self.source_len,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                add_special_tokens=True,
                                                return_tensors='pt')
        
        target = self.tokenizer.batch_encode_plus([whitespace_handler(summary)],
                                                max_length = self.target_len,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                add_special_tokens=True,
                                                return_tensors='pt')
        
        labels = target['input_ids']
        labels[labels == 0] = -100
        
        return (source['input_ids'].squeeze(),
                source['attention_mask'].squeeze(),
                labels.squeeze(),
                target['attention_mask'].squeeze())