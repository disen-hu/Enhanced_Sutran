import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, texts):
        # 收集所有独特的字符
        chars = set()
        for text in texts:
            chars.update(text)
        
        # 添加特殊token
        special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        self.token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        
        # 添加所有字符到词表
        for char in sorted(chars):
            self.token_to_id[char] = len(self.token_to_id)
            
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        
        # 特殊token的id
        self.pad_token_id = self.token_to_id['[PAD]']
        self.unk_token_id = self.token_to_id['[UNK]']
        self.bos_token_id = self.token_to_id['[BOS]']
        self.eos_token_id = self.token_to_id['[EOS]']
        
    def encode(self, text, max_length=None, padding=True, return_tensors=None):
        # 转换文本为id
        tokens = ['[BOS]'] + list(text) + ['[EOS]']
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            elif padding and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
                
        if return_tensors == 'pt':
            ids = torch.tensor(ids)
            
        return ids
    
    def decode(self, ids):
        if hasattr(ids, 'tolist'):  # 如果是tensor
            ids = ids.tolist()
            
        tokens = []
        for id_ in ids:
            token = self.id_to_token.get(id_, '[UNK]')
            if token not in ['[PAD]', '[BOS]', '[EOS]', '[UNK]']:
                tokens.append(token)
                
        return ''.join(tokens)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = []
        for text in texts:
            encoding = tokenizer.encode(
                text,
                max_length=max_length,
                padding=True,
                return_tensors='pt'
            )
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        input_ids = self.encodings[idx]
        
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank=None, dt_min=0.001, dt_max=0.1, dt_init='random', dt_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Initialize with smaller values to prevent explosion
        self.A = nn.Parameter(torch.randn(self.d_state, self.d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(self.d_state, self.d_inner) * 0.01)
        self.D = nn.Parameter(torch.zeros(self.d_inner))
        
        self.dt_projs = nn.Parameter(torch.randn(self.d_inner, self.dt_rank) * 0.01)
        
        if dt_init == 'random':
            log_dt = torch.rand(self.dt_rank) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        else:
            log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), self.dt_rank)
        
        self.log_dt = nn.Parameter(log_dt * dt_scale)

    def forward(self, u):
        batch_size, seq_len, _ = u.shape
        
        delta = torch.exp(self.dt_projs @ self.log_dt.exp() + 1e-6)
        delta = torch.clamp(delta, min=1e-6, max=10.0)
        
        dA = torch.matrix_exp(self.A * delta.mean())
        
        x = torch.zeros(batch_size, self.d_state, device=u.device)
        ys = []
        
        for t in range(seq_len):
            u_t = u[:, t]
            x = x @ dA + u_t @ self.B
            y = x @ self.C + u_t * self.D
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y / (seq_len ** 0.5)
        
        return y

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.zeros_(self.in_proj.bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            dt_rank=dt_rank,
        )
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        
        x = self.conv1d(x.transpose(-1, -2))[:, :, :-self.d_conv+1].transpose(-1, -2)
        x = self.ssm(x)
        
        gate = F.silu(gate)
        gate = torch.clamp(gate, min=-3, max=3)
        x = x * gate
        
        x = self.out_proj(x)
        x = (x + residual) / 2.0
        
        return x

class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layer=4, d_state=16, expand=2, d_conv=4, dt_rank=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model) * 0.01)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank
            )
            for _ in range(n_layer)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1)]
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

def generate_text(model, tokenizer, prefix, max_length=50, temperature=0.7, top_k=30, repetition_penalty=1.2):
    model.eval()
    device = next(model.parameters()).device
    
    # 确保最小生成长度
    min_length = 10
    
    input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    generated = []
    past_tokens = set()
    
    with torch.no_grad():
        try:
            for i in range(max_length):
                outputs = model(input_ids)
                next_token_logits = outputs[:, -1, :].clone()
                
                # 在最小长度之前，禁止生成句号等终止符号
                if i < min_length:
                    for end_token in ['。', '！', '？', '，']:
                        if end_token in tokenizer.token_to_id:
                            next_token_logits[0, tokenizer.token_to_id[end_token]] = -float('inf')
                
                # 应用重复惩罚
                for token_id in past_tokens:
                    next_token_logits[0, token_id] /= repetition_penalty
                
                # 过滤特殊token
                special_tokens_mask = torch.full((next_token_logits.shape[-1],), 0, device=device)
                special_tokens_list = [tokenizer.pad_token_id, tokenizer.unk_token_id, 
                                    tokenizer.bos_token_id, tokenizer.eos_token_id]
                for token_id in special_tokens_list:
                    special_tokens_mask[token_id] = -float('inf')
                next_token_logits = next_token_logits + special_tokens_mask
                
                # 应用temperature
                next_token_logits = next_token_logits / (temperature + 1e-6)
                
                # Top-k 采样
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # 采样下一个token
                probs = F.softmax(next_token_logits + 1e-6, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 获取生成的token
                token = tokenizer.decode([next_token.squeeze().item()])
                
                # 检查是否陷入重复
                if len(past_tokens) > 5 and len(set(list(past_tokens)[-5:])) < 2:
                    if len(generated) >= min_length:
                        break
                
                # 检查是否生成了终止符号
                if i >= min_length and token in ['。', '！', '？']:
                    generated.append(next_token.squeeze().item())
                    break
                
                generated.append(next_token.squeeze().item())
                past_tokens.add(next_token.squeeze().item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return prefix
    
    # 确保生成的文本有合适的终止符
    generated_text = tokenizer.decode(generated)
    if not generated_text.endswith(('。', '！', '？')):
        if len(generated_text) >= min_length:
            generated_text += '。'
        else:
            # 如果文本太短，继续生成直到达到最小长度
            generated_text = prefix + "技术在不断发展，带来新的机遇。"
    
    return generated_text

def get_sample_texts():
    return [
        "人工智能技术正在快速发展，为各行各业带来革命性的变化！",
        "人工智能在医疗领域的应用前景广阔，可以辅助医生诊断疾病。",
        "人工智能技术推动着自动驾驶汽车的发展，提高了交通安全性。",
        "智能机器人在工业生产中的应用越来越广泛，提高了生产效率。",
        "深度学习技术在图像识别领域取得了突破性进展，准确率不断提高！",
        "自然语言处理技术让机器能够更好地理解和生成人类语言！",
        "机器学习算法能够从大量数据中自动学习规律和模式，令人惊叹。",
        "计算机视觉技术让机器能够理解和分析图像与视频内容，应用广泛。",
        "语音识别技术使得人机交互变得更加自然和便捷，深受欢迎。",
        "推荐系统能够根据用户喜好精准推送个性化内容，提升体验！",
        "人工智能辅助教育让学习变得更加个性化和高效，激发学习兴趣！",
        "智能家居系统提供了更舒适和便利的生活体验，深受用户喜爱。",
        "人工智能在金融领域的应用提高了风险控制能力，意义重大！",
        "智能客服系统全天候为用户提供贴心服务，效率显著提升。",
        "人工智能技术在科学研究中发挥着重要作用，推动创新发展！",
        "机器翻译技术帮助打破了语言交流的障碍，促进文化交流。",
        "智能制造技术推动工业生产的自动化和智能化，引领变革？",
        "人工智能在农业领域的应用显著提高了生产效率，成果丰硕。",
        "智能物流系统优化了供应链管理和配送效率，带来便利！",
        "人工智能技术助力环境保护和资源节约，贡献巨大。"
    ]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sample_texts = get_sample_texts()
    
    # 扩充训练数据
    expanded_texts = []
    for text in sample_texts:
        expanded_texts.append(text)
        # 从句子中间截取子句也作为训练数据
        parts = text.split('，')
        if len(parts) > 1:
            for part in parts:
                if len(part.strip()) > 10:  # 只添加足够长的子句
                    # 随机选择结束符号
                    end_marks = ['。', '！', '？']
                    expanded_texts.append(part.strip() + end_marks[len(expanded_texts) % 3])
    
    # 初始化tokenizer和数据集
    tokenizer = CharTokenizer(expanded_texts)
    dataset = TextDataset(expanded_texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of training samples: {len(expanded_texts)}")
    
    # 初始化模型
    model = MambaModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layer=4,
        d_state=16,
        expand=2,
        d_conv=4,
        dt_rank=None
    ).to(device)
    
    # 优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, eps=1e-8)
    
    # 学习率预热
    warmup_steps = 200
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.1 ** (step / 2000)  # 更慢的学习率衰减
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练参数
    num_epochs = 10
    best_loss = float('inf')
    patience = 50
    no_improve_count = 0
    gradient_accumulation_steps = 4
    min_epochs = 100  # 最小训练轮数
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = loss_fct(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                optimizer.zero_grad()
                continue
        
        avg_loss = total_loss / num_batches
        
        # Early stopping 检查
        if epoch >= min_epochs:  # 只在最小轮数之后才考虑早停
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n测试生成:")
    test_prefixes = [
        "人工智能",
        "人工智能技术正在",
        "人工智能技术"
    ]
    
    model.eval()
    for prefix in test_prefixes:
        try:
            for _ in range(3):  # 对每个前缀生成3个不同的结果
                generated = generate_text(
                    model, 
                    tokenizer, 
                    prefix, 
                    max_length=50,
                    temperature=0.7,
                    top_k=30,
                    repetition_penalty=1.2
                )
                print(f"\n前缀: {prefix}")
                print(f"生成: {generated}")
        except Exception as e:
            print(f"生成失败: {str(e)}")

if __name__ == "__main__":
    main()