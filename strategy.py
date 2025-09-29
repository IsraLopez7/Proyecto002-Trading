# strategy.py
"""
Módulo de Estrategia de Trading
Implementa la lógica de trading con gestión de posiciones y riesgo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PositionType(Enum):
    """Tipos de posición"""
    NONE = 0
    LONG = 1
    SHORT = -1

@dataclass
class Position:
    """Clase para representar una posición de trading"""
    entry_date: pd.Timestamp
    entry_price: float
    position_type: PositionType
    size: float
    stop_loss: float
    take_profit: float
    entry_signal_strength: float = 0
    
    def calculate_pnl(self, current_price: float, commission: float = 0.00125) -> float:
        """Calcula el P&L de la posición"""
        if self.position_type == PositionType.LONG:
            gross_pnl = (current_price - self.entry_price) * self.size
        else:  # SHORT
            gross_pnl = (self.entry_price - current_price) * self.size
        
        # Descontar comisiones (entrada + salida)
        commission_cost = (self.entry_price * self.size + current_price * self.size) * commission
        return gross_pnl - commission_cost
    
    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """Determina si la posición debe cerrarse"""
        if self.position_type == PositionType.LONG:
            if current_price <= self.stop_loss:
                return True, "stop_loss"
            elif current_price >= self.take_profit:
                return True, "take_profit"
        else:  # SHORT
            if current_price >= self.stop_loss:
                return True, "stop_loss"
            elif current_price <= self.take_profit:
                return True, "take_profit"
        
        return False, ""

class TradingStrategy:
    """
    Clase principal de la estrategia de trading
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa la estrategia
        
        Parameters:
        -----------
        config : Dict
            Configuración del proyecto
        """
        self.config = config
        self.positions: List[Position] = []
        self.closed_positions: List[Dict] = []
        self.capital = config['trading']['initial_capital']
        self.available_capital = self.capital
        self.commission = config['trading']['commission']
        self.allow_short = config['trading']['allow_short']
        self.max_positions = config['risk_management']['max_positions']
        self.risk_per_trade = config['risk_management']['risk_per_trade']
        
        # Tracking de rendimiento
        self.equity_curve = []
        self.trade_history = []
        
    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """
        Calcula el tamaño de la posición basado en gestión de riesgo
        
        Parameters:
        -----------
        price : float
            Precio actual
        stop_loss : float
            Nivel de stop loss
            
        Returns:
        --------
        float
            Tamaño de la posición
        """
        # Riesgo máximo por operación
        risk_amount = self.available_capital * self.risk_per_trade
        
        # Distancia al stop loss
        stop_distance = abs(price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        # Tamaño basado en riesgo
        position_size = risk_amount / stop_distance
        
        # Limitar al capital disponible
        max_size = self.available_capital / price
        position_size = min(position_size, max_size * 0.95)  # 95% del capital disponible
        
        return position_size
    
    def open_position(self, date: pd.Timestamp, price: float, signal: int,
                     stop_loss: float, take_profit: float, 
                     signal_strength: float = 0) -> Optional[Position]:
        """
        Abre una nueva posición
        
        Parameters:
        -----------
        date : pd.Timestamp
            Fecha de entrada
        price : float
            Precio de entrada
        signal : int
            Señal de trading (1: long, -1: short)
        stop_loss : float
            Nivel de stop loss
        take_profit : float
            Nivel de take profit
        signal_strength : float
            Fuerza de la señal
            
        Returns:
        --------
        Position or None
            La posición creada o None si no se pudo abrir
        """
        # Verificar si podemos abrir más posiciones
        if len(self.positions) >= self.max_positions:
            return None
        
        # Verificar señal válida
        if signal == 0:
            return None
        
        # No permitir shorts si está deshabilitado
        if signal == -1 and not self.allow_short:
            return None
        
        # Calcular tamaño de posición
        position_size = self.calculate_position_size(price, stop_loss)
        
        if position_size <= 0:
            return None
        
        # Crear posición
        position = Position(
            entry_date=date,
            entry_price=price,
            position_type=PositionType.LONG if signal > 0 else PositionType.SHORT,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_signal_strength=signal_strength
        )
        
        # Actualizar capital disponible
        capital_used = position_size * price * (1 + self.commission)
        self.available_capital -= capital_used
        
        # Agregar a posiciones activas
        self.positions.append(position)
        
        return position
    
    def close_position(self, position: Position, date: pd.Timestamp, 
                       price: float, reason: str) -> Dict:
        """
        Cierra una posición
        
        Parameters:
        -----------
        position : Position
            Posición a cerrar
        date : pd.Timestamp
            Fecha de cierre
        price : float
            Precio de cierre
        reason : str
            Razón del cierre
            
        Returns:
        --------
        Dict
            Información del trade cerrado
        """
        # Calcular P&L
        pnl = position.calculate_pnl(price, self.commission)
        
        # Calcular retorno porcentual
        if position.position_type == PositionType.LONG:
            return_pct = ((price - position.entry_price) / position.entry_price) * 100
        else:
            return_pct = ((position.entry_price - price) / position.entry_price) * 100
        
        # Información del trade
        trade_info = {
            'entry_date': position.entry_date,
            'exit_date': date,
            'entry_price': position.entry_price,
            'exit_price': price,
            'position_type': 'LONG' if position.position_type == PositionType.LONG else 'SHORT',
            'size': position.size,
            'pnl': pnl,
            'return_pct': return_pct,
            'reason': reason,
            'holding_period': (date - position.entry_date).total_seconds() / 3600,  # en horas
            'signal_strength': position.entry_signal_strength
        }
        
        # Actualizar capital disponible
        capital_returned = position.size * price * (1 - self.commission)
        self.available_capital += capital_returned
        
        # Remover de posiciones activas
        self.positions.remove(position)
        
        # Agregar a historial
        self.closed_positions.append(trade_info)
        self.trade_history.append(trade_info)
        
        return trade_info
    
    def check_positions(self, date: pd.Timestamp, price: float) -> List[Dict]:
        """
        Verifica todas las posiciones activas
        
        Parameters:
        -----------
        date : pd.Timestamp
            Fecha actual
        price : float
            Precio actual
            
        Returns:
        --------
        List[Dict]
            Lista de trades cerrados
        """
        closed_trades = []
        positions_to_close = []
        
        for position in self.positions:
            should_close, reason = position.should_close(price)
            if should_close:
                positions_to_close.append((position, reason))
        
        # Cerrar posiciones
        for position, reason in positions_to_close:
            trade_info = self.close_position(position, date, price, reason)
            closed_trades.append(trade_info)
        
        return closed_trades
    
    def process_signal(self, date: pd.Timestamp, price: float, signal: int,
                       stop_loss: float, take_profit: float,
                       signal_strength: float = 0) -> Dict:
        """
        Procesa una señal de trading
        
        Parameters:
        -----------
        date : pd.Timestamp
            Fecha actual
        price : float
            Precio actual
        signal : int
            Señal de trading
        stop_loss : float
            Stop loss
        take_profit : float
            Take profit
        signal_strength : float
            Fuerza de la señal
            
        Returns:
        --------
        Dict
            Resultado del procesamiento
        """
        result = {
            'date': date,
            'price': price,
            'action': 'hold',
            'positions_opened': 0,
            'positions_closed': 0,
            'current_positions': len(self.positions)
        }
        
        # Primero verificar posiciones existentes
        closed_trades = self.check_positions(date, price)
        result['positions_closed'] = len(closed_trades)
        
        # Si hay señal y no tenemos posición, intentar abrir
        if signal != 0 and len(self.positions) < self.max_positions:
            # Verificar si ya tenemos una posición del mismo tipo
            has_same_position = any(
                (p.position_type == PositionType.LONG and signal > 0) or
                (p.position_type == PositionType.SHORT and signal < 0)
                for p in self.positions
            )
            
            if not has_same_position:
                position = self.open_position(
                    date, price, signal, stop_loss, take_profit, signal_strength
                )
                if position:
                    result['action'] = 'buy' if signal > 0 else 'sell'
                    result['positions_opened'] = 1
        
        # Si tenemos posición contraria a la señal, cerrarla
        for position in list(self.positions):
            if (position.position_type == PositionType.LONG and signal < 0) or \
               (position.position_type == PositionType.SHORT and signal > 0):
                trade_info = self.close_position(position, date, price, 'signal_reversal')
                result['positions_closed'] += 1
        
        result['current_positions'] = len(self.positions)
        return result
    
    def calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calcula el valor actual del portfolio
        
        Parameters:
        -----------
        current_price : float
            Precio actual del activo
            
        Returns:
        --------
        float
            Valor total del portfolio
        """
        # Capital disponible
        total = self.available_capital
        
        # Valor de las posiciones abiertas
        for position in self.positions:
            position_value = position.size * current_price
            if position.position_type == PositionType.SHORT:
                # Para shorts, el valor es la diferencia
                position_value = 2 * position.size * position.entry_price - position_value
            total += position_value
        
        return total
    
    def get_performance_summary(self) -> Dict:
        """
        Obtiene un resumen del rendimiento
        
        Returns:
        --------
        Dict
            Resumen de rendimiento
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'total_return': 0
            }
        
        trades_df = pd.DataFrame(self.trade_history)
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        summary = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
            'total_pnl': trades_df['pnl'].sum(),
            'total_return': (trades_df['pnl'].sum() / self.config['trading']['initial_capital']) * 100,
            'avg_holding_period': trades_df['holding_period'].mean() if len(trades_df) > 0 else 0,
            'max_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            'max_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
        }
        
        # Análisis por tipo de posición
        long_trades = trades_df[trades_df['position_type'] == 'LONG']
        short_trades = trades_df[trades_df['position_type'] == 'SHORT']
        
        summary['long_trades'] = {
            'total': len(long_trades),
            'wins': len(long_trades[long_trades['pnl'] > 0]),
            'win_rate': len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
            'total_pnl': long_trades['pnl'].sum() if len(long_trades) > 0 else 0
        }
        
        summary['short_trades'] = {
            'total': len(short_trades),
            'wins': len(short_trades[short_trades['pnl'] > 0]),
            'win_rate': len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
            'total_pnl': short_trades['pnl'].sum() if len(short_trades) > 0 else 0
        }
        
        return summary


class StrategyValidator:
    """
    Clase para validar y analizar la estrategia
    """
    
    def __init__(self, strategy: TradingStrategy):
        """
        Inicializa el validador
        
        Parameters:
        -----------
        strategy : TradingStrategy
            Estrategia a validar
        """
        self.strategy = strategy
        
    def analyze_trade_distribution(self) -> Dict:
        """
        Analiza la distribución de trades
        
        Returns:
        --------
        Dict
            Análisis de distribución
        """
        if not self.strategy.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.strategy.trade_history)
        
        analysis = {
            'trades_by_reason': trades_df['reason'].value_counts().to_dict(),
            'avg_pnl_by_reason': trades_df.groupby('reason')['pnl'].mean().to_dict(),
            'trades_by_signal_strength': {}
        }
        
        # Analizar por fuerza de señal
        if 'signal_strength' in trades_df.columns:
            trades_df['strength_category'] = pd.cut(
                trades_df['signal_strength'], 
                bins=[0, 2, 3, 5], 
                labels=['weak', 'medium', 'strong']
            )
            
            for category in ['weak', 'medium', 'strong']:
                cat_trades = trades_df[trades_df['strength_category'] == category]
                if len(cat_trades) > 0:
                    analysis['trades_by_signal_strength'][category] = {
                        'count': len(cat_trades),
                        'win_rate': len(cat_trades[cat_trades['pnl'] > 0]) / len(cat_trades) * 100,
                        'avg_pnl': cat_trades['pnl'].mean()
                    }
        
        return analysis
    
    def calculate_risk_metrics(self, equity_curve: List[float]) -> Dict:
        """
        Calcula métricas de riesgo
        
        Parameters:
        -----------
        equity_curve : List[float]
            Curva de equity
            
        Returns:
        --------
        Dict
            Métricas de riesgo
        """
        if len(equity_curve) < 2:
            return {}
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Calcular drawdown
        cumulative = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cumulative) / cumulative
        max_drawdown = np.min(drawdown)
        
        # Duración del drawdown
        drawdown_start = np.argmax(cumulative[:np.argmin(drawdown)]) if np.argmin(drawdown) > 0 else 0
        drawdown_end = np.argmin(drawdown)
        drawdown_duration = drawdown_end - drawdown_start
        
        metrics = {
            'max_drawdown': max_drawdown * 100,
            'drawdown_duration': drawdown_duration,
            'var_95': np.percentile(returns, 5) * 100 if len(returns) > 0 else 0,
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]) * 100 if len(returns) > 0 else 0,
            'volatility': np.std(returns) * np.sqrt(365 * 24) * 100 if len(returns) > 0 else 0,
            'downside_deviation': np.std(returns[returns < 0]) * np.sqrt(365 * 24) * 100 if len(returns[returns < 0]) > 0 else 0
        }
        
        return metrics


# Función de prueba
def test_strategy(config, df_with_signals):
    """
    Prueba la estrategia de trading
    """
    strategy = TradingStrategy(config)
    
    print("\n=== Ejecutando Backtest de Estrategia ===")
    print(f"Capital inicial: ${config['trading']['initial_capital']:,}")
    print(f"Comisión: {config['trading']['commission']*100}%")
    print(f"Posiciones cortas: {'Sí' if config['trading']['allow_short'] else 'No'}")
    
    # Variables para tracking
    equity_curve = [config['trading']['initial_capital']]
    
    # Simular trading
    for i in range(len(df_with_signals)):
        row = df_with_signals.iloc[i]
        
        # Solo procesar si tenemos todos los indicadores calculados
        if pd.isna(row.get('combined_signal', np.nan)):
            continue
        
        # Obtener señales y niveles
        signal = row.get('combined_signal', 0)
        signal_strength = row.get('signal_strength', 0)
        stop_loss = row.get('SL_long' if signal > 0 else 'SL_short', row['Close'] * 0.98)
        take_profit = row.get('TP_long' if signal > 0 else 'TP_short', row['Close'] * 1.02)
        
        # Procesar señal
        result = strategy.process_signal(
            date=row['Date'],
            price=row['Close'],
            signal=signal,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_strength=signal_strength
        )
        
        # Actualizar equity curve
        portfolio_value = strategy.calculate_portfolio_value(row['Close'])
        equity_curve.append(portfolio_value)
    
    # Cerrar posiciones restantes
    if len(strategy.positions) > 0:
        last_row = df_with_signals.iloc[-1]
        for position in list(strategy.positions):
            strategy.close_position(position, last_row['Date'], last_row['Close'], 'end_of_period')
    
    # Obtener resumen
    performance = strategy.get_performance_summary()
    
    print("\n=== Resumen de Performance ===")
    print(f"Total de trades: {performance['total_trades']}")
    print(f"Trades ganadores: {performance['winning_trades']}")
    print(f"Trades perdedores: {performance['losing_trades']}")
    print(f"Win Rate: {performance['win_rate']:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"P&L Total: ${performance['total_pnl']:.2f}")
    print(f"Retorno Total: {performance['total_return']:.2f}%")
    
    # Análisis adicional
    validator = StrategyValidator(strategy)
    risk_metrics = validator.calculate_risk_metrics(equity_curve)
    
    if risk_metrics:
        print("\n=== Métricas de Riesgo ===")
        print(f"Max Drawdown: {risk_metrics['max_drawdown']:.2f}%")
        print(f"Volatilidad Anualizada: {risk_metrics['volatility']:.2f}%")
        print(f"VaR 95%: {risk_metrics['var_95']:.2f}%")
    
    return strategy, equity_curve


if __name__ == "__main__":
    # Código de prueba
    pass