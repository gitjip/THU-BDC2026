import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
from utils import apply_market_relative_features
from utils import apply_market_breadth_features
from utils import apply_market_env_features
from utils import apply_trend_quality_features
from utils import apply_cross_sectional_rank_features
from utils import apply_rank_momentum_features
from utils import apply_rank_riskadj_features
from utils import apply_ret5_rank_features
from utils import apply_short_overheat_features
from utils import apply_clean_risk_features
from utils import apply_multi_period_features
import joblib
import os
import json
import multiprocessing as mp
import random
import logging
import sys
import time
from data_utils import load_stock_data, setup_logging, split_train_val_by_trading_days

logger = logging.getLogger(__name__)


def format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def log_section(title):
    line = "=" * 72
    logger.info("\n%s\n%s\n%s", line, title, line)


def show_progress_bar():
    return sys.stderr.isatty()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_cloums_map = {
    '39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['instrument','开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def select_feature_columns(feature_columns):
    if config.get('use_instrument_feature', True):
        return list(feature_columns)
    return [column for column in feature_columns if column != 'instrument']


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed = processed.copy()
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-config.get('label_horizon', 5))

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed.loc[processed['open_t1'] > 1e-4].copy()

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label']).copy()

    return processed.drop(columns=['open_t1', 'open_t5'])


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    if config['feature_num'] not in feature_engineer_func_map:
        raise ValueError(f"Unsupported feature_num: {config['feature_num']}")
    if stockid2idx is None:
        raise ValueError("stockid2idx 不能为空")
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = select_feature_columns(feature_cloums_map[config['feature_num']])

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(config.get('num_processes', 10), mp.cpu_count(), len(groups))
    logger.info("开始%s: 股票数=%s, 进程数=%s", desc, len(groups), num_processes)
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc, disable=not show_progress_bar()))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    if config.get('use_market_relative_features', False):
        processed, feature_columns, market_relative_columns = apply_market_relative_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加市场相对特征 %s 个，输入特征=%s",
            desc,
            len(market_relative_columns),
            len(feature_columns),
        )

    if config.get('use_market_breadth_features', False):
        processed, feature_columns, market_breadth_columns = apply_market_breadth_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加市场宽度特征 %s 个，输入特征=%s",
            desc,
            len(market_breadth_columns),
            len(feature_columns),
        )

    if config.get('use_market_env_features', False):
        market_env_feature_set = config.get('market_env_feature_set', 'full')
        processed, feature_columns, market_env_columns = apply_market_env_features(
            processed,
            feature_columns,
            feature_set=market_env_feature_set,
        )
        logger.info(
            "%s: 已添加市场环境特征 set=%s, 列=%s 个，输入特征=%s",
            desc,
            market_env_feature_set,
            len(market_env_columns),
            len(feature_columns),
        )

    if config.get('use_trend_quality_features', False):
        processed, feature_columns, trend_quality_columns = apply_trend_quality_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加趋势质量特征 %s 个，输入特征=%s",
            desc,
            len(trend_quality_columns),
            len(feature_columns),
        )

    if config.get('use_cross_sectional_rank_features', False):
        rank_mode = config.get('cross_sectional_rank_mode', 'append')
        rank_replace_set = config.get('cross_sectional_rank_replace_set', 'default')
        processed, feature_columns, rank_feature_columns = apply_cross_sectional_rank_features(
            processed,
            feature_columns,
            mode=rank_mode,
            replace_set=rank_replace_set,
        )
        logger.info(
            "%s: 已应用横截面 rank 特征 mode=%s, replace_set=%s, rank列=%s, 输入特征=%s",
            desc,
            rank_mode,
            rank_replace_set,
            len(rank_feature_columns),
            len(feature_columns),
        )

    if config.get('use_rank_momentum_features', False):
        processed, feature_columns, rank_momentum_columns = apply_rank_momentum_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加横截面动量变化特征 %s 个，输入特征=%s",
            desc,
            len(rank_momentum_columns),
            len(feature_columns),
        )

    if config.get('use_rank_riskadj_features', False):
        processed, feature_columns, rank_riskadj_columns = apply_rank_riskadj_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加横截面风险调整动量特征 %s 个，输入特征=%s",
            desc,
            len(rank_riskadj_columns),
            len(feature_columns),
        )

    if config.get('use_ret5_rank_features', False):
        processed, feature_columns, ret5_rank_columns = apply_ret5_rank_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加 return_5 横截面 rank 特征 %s 个，输入特征=%s",
            desc,
            len(ret5_rank_columns),
            len(feature_columns),
        )

    if config.get('use_short_overheat_features', False):
        processed, feature_columns, short_overheat_columns = apply_short_overheat_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加短期过热保护特征 %s 个，输入特征=%s",
            desc,
            len(short_overheat_columns),
            len(feature_columns),
        )

    if config.get('use_multi_period_features', False):
        processed, feature_columns, multi_period_columns = apply_multi_period_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加多周期基础特征 %s 个，输入特征=%s",
            desc,
            len(multi_period_columns),
            len(feature_columns),
        )

    if config.get('use_clean_risk_features', False):
        processed, feature_columns, clean_risk_columns = apply_clean_risk_features(
            processed,
            feature_columns,
        )
        logger.info(
            "%s: 已添加清洗启发的风险特征 %s 个，输入特征=%s",
            desc,
            len(clean_risk_columns),
            len(feature_columns),
        )

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    if not config.get('use_instrument_feature', True):
        logger.info("%s: 已从输入特征中移除 instrument，股票代码仅用于分组和输出", desc)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# 加权的排序损失函数
class WeightedRankingLoss(nn.Module):
    """
    组合的加权排序损失函数，着重强调top-k的样本。
    """
    def __init__(self, temperature=1.0, target_temperature=None, k=5, weight_factor=2.0, pairwise_weight=1, base_weight=1.0):
        super(WeightedRankingLoss, self).__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, current: {temperature}")
        if target_temperature is None:
            target_temperature = temperature
        if target_temperature <= 0:
            raise ValueError(f"target_temperature must be positive, current: {target_temperature}")
        self.temperature = temperature
        self.target_temperature = target_temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        """加权的Listwise损失 (KL散度 + Cross Entropy)"""
        
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.target_temperature, dim=1)

        # 加权 Cross Entropy（原实现未使用 weights）
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()
        
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        """加权的Pairwise损失"""
        batch_size, num_items = y_pred.size()
        
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        
        # 只考虑真实标签不同的项目对
        mask = (true_diff != 0).float()
        
        # 创建权重矩阵
        # 如果一对(i, j)中，i或j是关键样本，则权重更高
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        # weight_matrix = torch.where(weight_matrix > 2.0, self.weight_factor, 1.0)
        
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        
        # 应用mask和权重
        weighted_loss = pairwise_loss * mask * weight_matrix
        
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        
        return loss
        
    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items] (真实涨跌幅)
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        # 1. 识别 top-k 的样本
        _, top_indices = torch.topk(y_true, k, dim=1)
        
        # 2. 创建权重向量
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor
            
        # 3. 计算加权损失
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        
        # 组合两种损失
        total_loss = listwise + self.pairwise_weight * pairwise
        
        return total_loss

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算新的评估指标：Top 5 收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        
        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()
        
        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(self.targets[idx]),      # [num_stocks] 真实涨跌幅
            'relevance': torch.LongTensor(self.relevance_scores[idx]),  # [num_stocks] 排序标签
            'stock_indices': torch.LongTensor(self.stock_indices[idx])  # [num_stocks] 股票索引
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }


def calculate_grad_norm(parameters):
    total_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    grad_norm_total = 0.0
    grad_norm_steps = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}", disable=not show_progress_bar()):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # 模型预测
        outputs = model(sequences)  # [batch, max_stocks] 预测分数
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)
        
        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            
            if valid_indices.numel() == 0:
                continue
                
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            # 获取有效股票的预测值和预处理好的相关性得分
            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]
            
            if len(valid_pred) > 1:
                # 直接使用预处理好的相关性得分，无需重新计算
                loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            if config.get('enable_grad_clip', True):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            else:
                grad_norm_value = calculate_grad_norm(model.parameters())
            grad_norm_total += grad_norm_value
            grad_norm_steps += 1
            if writer:
                writer.add_scalar('train/grad_norm', grad_norm_value, global_step=epoch*len(dataloader)+local_step)
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    avg_grad_norm = grad_norm_total / grad_norm_steps if grad_norm_steps > 0 else 0.0
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics, avg_grad_norm

def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}", disable=not show_progress_bar()):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            # 模型预测
            outputs = model(sequences)
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                
                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()
                    
                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


class LookaheadOptimizer:
    """Lightweight Lookahead wrapper for an existing PyTorch optimizer."""

    def __init__(self, optimizer, k=5, alpha=0.5):
        if k < 1:
            raise ValueError("lookahead_k must be >= 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("lookahead_alpha must be in (0, 1]")
        self.optimizer = optimizer
        self.k = int(k)
        self.alpha = float(alpha)
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self._step_count = 0
        self._slow_weights = [
            [param.detach().clone() for param in group['params']]
            for group in self.param_groups
        ]
        for group in self._slow_weights:
            for slow_weight in group:
                slow_weight.requires_grad = False

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        loss = self.optimizer.step()
        self._step_count += 1
        if self._step_count % self.k == 0:
            for group, slow_group in zip(self.param_groups, self._slow_weights):
                for param, slow_weight in zip(group['params'], slow_group):
                    if param.grad is None:
                        continue
                    slow_weight.add_(param.data - slow_weight, alpha=self.alpha)
                    param.data.copy_(slow_weight)
        return loss

    def state_dict(self):
        state = self.optimizer.state_dict()
        state['lookahead'] = {
            'k': self.k,
            'alpha': self.alpha,
            'step_count': self._step_count,
            'slow_weights': self._slow_weights,
        }
        return state

    def load_state_dict(self, state_dict):
        lookahead_state = state_dict.pop('lookahead', None)
        self.optimizer.load_state_dict(state_dict)
        if lookahead_state:
            self.k = int(lookahead_state.get('k', self.k))
            self.alpha = float(lookahead_state.get('alpha', self.alpha))
            self._step_count = int(lookahead_state.get('step_count', self._step_count))
            self._slow_weights = lookahead_state.get('slow_weights', self._slow_weights)


def build_optimizer(model):
    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5),
    )
    optimizer_type = str(config.get('optimizer', 'adamw')).strip().lower()
    if optimizer_type == 'adamw':
        return base_optimizer
    if optimizer_type == 'lookahead':
        return LookaheadOptimizer(
            base_optimizer,
            k=config.get('lookahead_k', 5),
            alpha=config.get('lookahead_alpha', 0.5),
        )
    raise ValueError(f"Unsupported optimizer: {config.get('optimizer')}")


def get_scheduler_optimizer(optimizer):
    return optimizer.optimizer if isinstance(optimizer, LookaheadOptimizer) else optimizer


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def build_lr_scheduler(optimizer):
    scheduler_type = str(config.get('lr_scheduler', 'plateau')).strip().lower()
    if scheduler_type in {'', 'none', 'off', 'false'}:
        return None

    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 1),
            min_lr=config.get('min_learning_rate', 1e-6),
            threshold=config.get('lr_threshold', 1e-4),
            threshold_mode='abs',
        )

    if scheduler_type == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=max(config.get('min_learning_rate', 1e-6) / config['learning_rate'], 0.01),
            total_iters=max(config['num_epochs'], 1),
        )

    raise ValueError(f"Unsupported lr_scheduler: {config.get('lr_scheduler')}")


def step_lr_scheduler(scheduler, final_score):
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(final_score)
    else:
        scheduler.step()


def log_runtime_environment(device):
    logger.info("训练设备: %s", device)
    if device.type == 'cuda':
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        logger.info(
            "CUDA 设备: %s | 显存=%.1fGB",
            torch.cuda.get_device_name(device_index),
            props.total_memory / (1024 ** 3),
        )
    else:
        logger.info("CUDA 不可用，训练会主要依赖 CPU；Intel Arc 通常不会被 PyTorch CUDA 自动使用")
    logger.info("CPU 核心数: %s", mp.cpu_count())


def write_training_history(output_dir, records):
    history_path = os.path.join(output_dir, 'training_history.csv')
    pd.DataFrame(records).to_csv(history_path, index=False)
    return history_path


def split_train_val_by_recent_trading_days(df, sequence_length):
    """按真实交易日划分验证集，并为验证集补足历史窗口上下文。"""
    return split_train_val_by_trading_days(
        df=df,
        sequence_length=sequence_length,
        val_days=config.get('val_days', 20),
        label_horizon=config.get('label_horizon', 5),
        train_target_days=config.get('train_target_days', 0),
        logger=logger,
    )

# 主程序
def main():
    global logger
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    setup_logging("bdc.train", os.path.join(output_dir, 'train.log'))
    logger = logging.getLogger("bdc.train")
    run_start_time = time.perf_counter()
    torch_num_threads = config.get('torch_num_threads', 0)
    if torch_num_threads and torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)

    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        f.write('\n')
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train and config.get('enable_tensorboard', True) else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    mode = 'debug' if config.get('fast_dev_mode') else 'full'
    logger.info("运行模式: %s", mode)
    log_runtime_environment(device)
    logger.info("随机种子: %s", config.get('seed', 42))
    if torch_num_threads and torch_num_threads > 0:
        logger.info("PyTorch CPU 线程数: %s", torch.get_num_threads())
    if config.get('train_target_days', 0) or config.get('max_stocks_per_day', 0):
        logger.info(
            "调试采样: train_target_days=%s, max_stocks_per_day=%s",
            config.get('train_target_days', 0),
            config.get('max_stocks_per_day', 0),
        )
    
    # 1. 数据加载
    full_df, data_file = load_stock_data(
        data_path,
        data_file=config.get('stock_data_file'),
        allow_train_fallback=True,
        logger=logger,
    )
    train_df, val_df, val_start, _ = split_train_val_by_recent_trading_days(
        full_df,
        config['sequence_length'],
    )
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)
    logger.info("股票映射数量: %s", num_stocks)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    logger.info("特征数量: %s", len(features))
    
    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    if train_data.empty or val_data.empty:
        raise ValueError("特征清洗后训练集或验证集为空")
    # 然后再缩放
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    logger.info("Scaler 已保存: %s", os.path.join(output_dir, 'scaler.pkl'))

    
    # 4. 创建排序数据集
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('train_ranking_data_path'),
        max_stocks_per_date=config.get('max_stocks_per_day', 0),
        stock_sample_seed=config.get('seed', 42),
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data,
        features,
        config['sequence_length'],
        ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d'),
        max_stocks_per_date=config.get('max_stocks_per_day', 0),
        stock_sample_seed=config.get('seed', 42),
    )

    logger.info("训练集样本数: %s", len(train_sequences))
    logger.info("验证集样本数: %s", len(val_sequences))
    if len(train_sequences) == 0 or len(val_sequences) == 0:
        raise ValueError("训练集或验证集排序样本为空，请检查数据时间范围和 sequence_length")
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    logger.info("模型参数量: %s", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=config.get('loss_temperature', 1.0),
        target_temperature=config.get('loss_target_temperature', config.get('loss_temperature', 1.0)),
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0)
    )  # 使用加权排序损失
    optimizer = build_optimizer(model)
    scheduler = build_lr_scheduler(get_scheduler_optimizer(optimizer))
    logger.info(
        "训练策略: epochs<=%s, lr=%s, weight_decay=%s, optimizer=%s, lookahead_k=%s, lookahead_alpha=%s, scheduler=%s, lr_patience=%s, lr_threshold=%s, early_stopping_patience=%s, grad_clip=%s, loss_temperature=%s, loss_target_temperature=%s",
        config['num_epochs'],
        config['learning_rate'],
        config.get('weight_decay', 1e-5),
        config.get('optimizer', 'adamw'),
        config.get('lookahead_k', 5),
        config.get('lookahead_alpha', 0.5),
        config.get('lr_scheduler', 'plateau'),
        config.get('lr_patience', 1),
        config.get('lr_threshold', 1e-4),
        config.get('early_stopping_patience', 0),
        config.get('enable_grad_clip', True),
        config.get('loss_temperature', 1.0),
        config.get('loss_target_temperature', config.get('loss_temperature', 1.0)),
    )
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        best_epoch = -1
        epochs_without_improvement = 0
        stopped_epoch = 0
        stop_reason = 'max_epochs'
        min_delta = config.get('early_stopping_min_delta', 1e-4)
        early_stopping_patience = config.get('early_stopping_patience', 0)
        history_records = []
        log_section("开始训练")
        
        for epoch in range(config['num_epochs']):
            epoch_number = epoch + 1
            epoch_start_time = time.perf_counter()
            log_section(f"Epoch {epoch_number}/{config['num_epochs']}")
            
            # 训练
            train_start_time = time.perf_counter()
            train_loss, train_metrics, avg_grad_norm = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )
            train_seconds = time.perf_counter() - train_start_time
            
            logger.info("Train Loss: %.4f | avg_grad_norm=%.6f | 耗时=%s", train_loss, avg_grad_norm, format_duration(train_seconds))
            for k, v in train_metrics.items():
                logger.info("Train %s: %.4f", k, v)
            
            # 验证
            eval_start_time = time.perf_counter()
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )
            eval_seconds = time.perf_counter() - eval_start_time
            
            logger.info("Eval Loss: %.4f | 耗时=%s", eval_loss, format_duration(eval_seconds))
            for k, v in eval_metrics.items():
                logger.info("Eval %s: %.4f", k, v)

            # 保存最佳模型（基于final score）
            current_final_score = eval_metrics.get('final_score', 0.0)
            improved = current_final_score > best_score + min_delta
            if improved:
                best_score = current_final_score
                best_epoch = epoch_number
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                logger.info("保存最佳模型: %s | final_score=%.4f", os.path.join(output_dir, 'best_model.pth'), best_score)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "验证 final_score 未明显提升: 当前=%.4f, 最佳=%.4f, 连续未提升=%s/%s",
                    current_final_score,
                    best_score,
                    epochs_without_improvement,
                    early_stopping_patience if early_stopping_patience else 'off',
                )

            previous_lr = get_current_lr(optimizer)
            step_lr_scheduler(scheduler, current_final_score)
            current_lr = get_current_lr(optimizer)
            if writer:
                writer.add_scalar('train/learning_rate', current_lr, global_step=epoch)
            if current_lr != previous_lr:
                logger.info("学习率调整: %.8f -> %.8f", previous_lr, current_lr)
            else:
                logger.info("当前学习率: %.8f", current_lr)

            epoch_seconds = time.perf_counter() - epoch_start_time
            stopped_epoch = epoch_number
            history_records.append(
                {
                    "epoch": epoch_number,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "train_final_score": train_metrics.get("final_score", 0.0),
                    "eval_final_score": current_final_score,
                    "learning_rate": current_lr,
                    "best_score": best_score,
                    "epochs_without_improvement": epochs_without_improvement,
                    "avg_grad_norm": avg_grad_norm,
                    "train_seconds": round(train_seconds, 3),
                    "eval_seconds": round(eval_seconds, 3),
                    "epoch_seconds": round(epoch_seconds, 3),
                    "improved": improved,
                }
            )
            history_path = write_training_history(output_dir, history_records)
            logger.info("Epoch %s 完成: 耗时=%s | history=%s", epoch_number, format_duration(epoch_seconds), history_path)

            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                stop_reason = f'early_stopping_patience_{early_stopping_patience}'
                logger.info(
                    "触发早停: 连续 %s 个 epoch 未超过最佳分数 %.4f + min_delta %.6f",
                    epochs_without_improvement,
                    best_score,
                    min_delta,
                )
                break

        total_seconds = time.perf_counter() - run_start_time
        final_lr = get_current_lr(optimizer)
        logger.info(
            "训练完成: 最佳 epoch=%s, 最佳 final_score=%.4f, 实际 epoch=%s, 结束原因=%s, 总耗时=%s",
            best_epoch,
            best_score,
            stopped_epoch,
            stop_reason,
            format_duration(total_seconds),
        )
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")
            f.write(f"Stopped epoch: {stopped_epoch}\nStop reason: {stop_reason}\n")
            f.write(f"Total duration: {format_duration(total_seconds)}\n")
            f.write(f"Total seconds: {total_seconds:.3f}\n")
            f.write(f"Final learning_rate: {final_lr:.10f}\n")
            f.write(f"Training history: {os.path.join(output_dir, 'training_history.csv')}\n")

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    logger.info("########## 训练完成！最佳 final score: %.4f ##########", best_score)
