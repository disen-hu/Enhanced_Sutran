import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np
from typing import Optional

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = []
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        input_ids = item['input_ids'].squeeze()
        attention_mask = item['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class CausalConv1d(nn.Module):
    def __init__(self, chan_in, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(chan_in, chan_in, kernel_size)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = CausalConv1d(self.d_inner, d_conv)
        
        self.A = nn.Parameter(torch.randn(self.d_state, self.d_state) / self.d_state ** 0.5)
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state) / self.d_state ** 0.5)
        self.C = nn.Parameter(torch.randn(self.d_state, self.d_inner) / self.d_state ** 0.5)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.layer_norm = nn.LayerNorm(d_model)  # 添加层归一化

    def forward(self, x, state=None):
        residual = x  # 添加残差连接
        batch_size, seq_len, _ = x.shape
        
        x_proj = self.in_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        
        x = self.conv1d(x.transpose(-1, -2)).transpose(-1, -2)
        
        if state is None:
            state = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        next_states = []
        
        for t in range(seq_len):
            new_state = state @ self.A + x[:, t] @ self.B
            y = new_state @ self.C + x[:, t] * self.D
            outputs.append(y)
            next_states.append(new_state)
            state = new_state
        
        x = torch.stack(outputs, dim=1)
        state = torch.stack(next_states, dim=1)[:, -1]
        
        x = x * torch.sigmoid(gate)
        x = self.out_proj(x)
        x = self.layer_norm(x + residual)  # 应用层归一化和残差连接
        
        return x, state

class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) / d_model ** 0.5)  # 添加位置编码
        self.dropout = nn.Dropout(0.1)  # 添加dropout
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, states=None):
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1)]  # 添加位置编码
        x = self.dropout(x)
        
        if states is None:
            states = [None] * len(self.layers)
        
        new_states = []
        for layer, prev_state in zip(self.layers, states):
            x, state = layer(x, prev_state)
            new_states.append(state)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_states

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, temperature=0.7):
    logits = logits / temperature
    
    # Top-k filtering
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(-1).expand_as(logits)
    filtered_logits = torch.where(logits < min_values, 
                                torch.ones_like(logits) * float('-inf'), 
                                logits)
    
    # Top-p filtering
    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    filtered_logits[indices_to_remove] = float('-inf')
    
    return filtered_logits

def generate_text(model, tokenizer, prefix, max_length=50, device='cuda', temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型输出
            outputs, _ = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # 使用 top-k 采样（取概率最高的前5个token）
            top_k = 5
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # 计算softmax概率
            exp_logits = torch.exp(top_k_logits)
            probs = exp_logits / exp_logits.sum()
            
            # 选择概率最高的token
            next_token_id = top_k_indices[0].item()
            
            # 如果生成了特殊token则停止
            if next_token_id == tokenizer.sep_token_id or next_token_id == tokenizer.pad_token_id:
                break
                
            # 添加到生成序列中
            generated_tokens.append(next_token_id)
            
            # 更新输入序列
            input_ids = torch.cat([input_ids, top_k_indices[0].unsqueeze(0).unsqueeze(0)], dim=1)
            
            # 检查长度
            if len(generated_tokens) >= max_length:
                break
    
    # 解码生成的token
    try:
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return prefix + generated_text
    except Exception as e:
        print(f"Error during decoding: {e}")
        return prefix

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 扩展训练样本
    sample_text = [
        "人工智能技术正在快速发展，为各行各业带来革命性的变化。",
        "人工智能在医疗领域的应用前景广阔，可以辅助医生诊断疾病。",
        "人工智能技术推动着自动驾驶汽车的发展，提高了交通安全性。",
        "智能机器人在工业生产中的应用越来越广泛，提高了生产效率。"
    ]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    dataset = TextDataset(sample_text, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = MambaModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layer=4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # 降低学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)  # 添加学习率调度器
    
    num_epochs = 1000  # 增加训练轮次
    best_loss = float('inf')
    patience = 20  # 早停的耐心值
    no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 添加梯度裁剪
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n生成测试:")
    test_prefixes = [
        "人工智能",
        "人工智能技术正在",
        "人工智能技术"
    ]
    
    for prefix in test_prefixes:
        print(f"\n前缀: {prefix}")
        try:
            # 使用较低的temperature和较短的生成长度
            generated = generate_text(
                model,
                tokenizer,
                prefix,
                max_length=20,
                device=device,
                temperature=0.6
            )
            print(f"生成结果: {generated}")
        except Exception as e:
            print(f"生成失败: {str(e)}")
        print("-" * 50)

if __name__ == "__main__":
    main()