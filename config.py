"""
Configuration module for Autonomous Self-Reinforcing Neural Trading Network.
Centralizes all configuration with environment-aware defaults.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
import logging

class TradingMode(Enum):
    """Trading operation modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class ModelUpdateStrategy(Enum):
    """Model update strategies"""
    CONTINUOUS = "continuous"
    BATCH = "batch"
    EPISODIC = "episodic"

@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    project_id: str = os.getenv("FIREBASE_PROJECT_ID", "trading-network-prod")
    credentials_path: str = os.getenv("FIREBASE_CREDENTIALS", "credentials/firebase-key.json")
    firestore_collection: str = os.getenv("FIRESTORE_COLLECTION", "trading_network")
    
    def validate(self) -> bool:
        """Validate Firebase configuration"""
        if not os.path.exists(self.credentials_path):
            logging.warning(f"Firebase credentials not found at {self.credentials_path}")
            return False
        return bool(self.project_id)

@dataclass
class ExchangeConfig:
    """Exchange API configuration"""
    exchange_name: str = os.getenv("EXCHANGE", "binance")
    api_key: Optional[str] = os.getenv("EXCHANGE_API_KEY")
    api_secret: Optional[str] = os.getenv("EXCHANGE_API_SECRET")
    testnet: bool = os.getenv("EXCHANGE_TESTNET", "True").lower() == "true"
    symbols: list = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT"]
    
    def get_ccxt_config(self) -> Dict[str, Any]:
        """Generate CCXT-compatible configuration"""
        config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        if self.testnet:
            config['options']['testnet'] = True
        return config

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    gamma: float = 0.99  # Discount factor
    learning_rate: float = 0.0001
    batch_size: int = 64
    buffer_size: int = 10000
    tau: float = 0.005  # Soft update parameter
    update_every: int = 100  # Update frequency
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    model_update_strategy: ModelUpdateStrategy = ModelUpdateStrategy.CONTINUOUS

@dataclass
class NeuralNetworkConfig:
    """Neural network architecture configuration"""
    input_size: int = 40  # Number of features
    hidden_layers: list = None
    output_size: int = 3  # Buy, Sell, Hold
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    initial_balance: float = 10000.0
    position_size_pct: float = 0.1  # 10% per trade
    max_position_pct: float = 0.3  # Max 30% in single position
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    trading_fee_pct: float = 0.001  # 0.1% trading fee
    mode: TradingMode = TradingMode.PAPER

@dataclass
class DataConfig:
    """Data processing configuration"""
    timeframe: str = "1h"
    lookback_window: int = 100
    technical_indicators: list = None
    normalize_features: bool = True
    
    def __post_init__(self):
        if self.technical_indicators is None:
            self.technical_indicators = [
                'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower',
                'volume', 'close', 'high', 'low', 'open'
            ]

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = os.getenv("LOG_FILE", "logs/trading_network.log")
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.firebase = FirebaseConfig()
        self.exchange = ExchangeConfig()
        self.rl = RLConfig()
        self.nn = NeuralNetworkConfig()
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        
        # Validate critical configurations
        self._validate()
    
    def _validate(self):
        """Validate all configurations"""
        validations = [
            ("Firebase", self.firebase.validate()),
            ("Exchange API Key", bool(self.exchange.api_key) if self.exchange.testnet else True),
        ]
        
        for name, condition in validations:
            if not condition:
                logging.error(f"Configuration validation failed for {name}")
        
        # Ensure log directory exists
        if self.logging.file_path:
            os.makedirs(os.path.dirname(self.logging.file_path), exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'trading_mode': self.trading.m