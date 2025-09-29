"""
Motor de Backtesting
Ejecuta simulaciones completas de la estrategia con métricas detalladas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from indicators import TechnicalIndicators
from strategy import TradingStrategy, StrategyValidator

class BacktestEngine:
    """
    Motor principal de backtesting
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa el motor de backtesting
        
        Parameters:
        -----------
        config : Dict
            Configuración del proyecto
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.indicators = TechnicalIndicators(config)
        self.strategy = None
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, indicator_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepara los datos con indicadores y señales
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con datos OHLCV
        indicator_params : Dict, optional
            Parámetros personalizados para indicadores
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con indicadores y señales
        """
        # Calcular indicadores
        df_with_indicators = self.indicators.calculate_all_indicators(df, indicator_params)
        
        # Generar señales combinadas
        df_with_signals = self.indicators.generate_combined_signal(df_with_indicators)
        
        return df_with_signals
    
    def run_backtest(self, df: pd.DataFrame, indicator_params: Optional[Dict] = None,
                    verbose: bool = True) -> Dict:
        """
        Ejecuta el backtest completo
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame con datos OHLCV
        indicator_params : Dict, optional
            Parámetros de indicadores
        verbose : bool
            Si imprimir información durante la ejecución
            
        Returns:
        --------
        Dict
            Resultados del backtest
        """
        if verbose:
            print("\n=== Iniciando Backtest ===")
            print(f"Período: {df['Date'].min().date()} a {df['Date'].max().date()}")
            print(f"Total de registros: {len(df)}")
        
        # Preparar datos con indicadores
        df_prepared = self.prepare_data(df, indicator_params)
        
        # Inicializar estrategia
        self.strategy = TradingStrategy(self.config)
        
        # Variables para tracking
        equity_curve = []
        portfolio_values = []
        positions_history = []
        trades_log = []
        
        # Ejecutar estrategia barra por barra
        for i in range(len(df_prepared)):
            row = df_prepared.iloc[i]
            
            # Skip si no hay señales calculadas todavía
            if pd.isna(row.get('combined_signal', np.nan)):
                equity_curve.append(self.strategy.capital)
                portfolio_values.append(self.strategy.capital)
                continue
            
            # Obtener parámetros de la señal
            signal = int(row.get('combined_signal', 0))
            signal_strength = row.get('signal_strength', 0)
            
            # Usar ATR para stop loss y take profit dinámicos
            if signal > 0:  # Long
                stop_loss = row.get('SL_long', row['Close'] * 0.98)
                take_profit = row.get('TP_long', row['Close'] * 1.03)
            elif signal < 0:  # Short
                stop_loss = row.get('SL_short', row['Close'] * 1.02)
                take_profit = row.get('TP_short', row['Close'] * 0.97)
            else:
                stop_loss = row['Close'] * 0.98
                take_profit = row['Close'] * 1.02
            
            # Procesar señal
            result = self.strategy.process_signal(
                date=row['Date'],
                price=row['Close'],
                signal=signal,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_strength=signal_strength
            )
            
            # Registrar acción
            if result['positions_opened'] > 0 or result['positions_closed'] > 0:
                trades_log.append({
                    'date': row['Date'],
                    'price': row['Close'],
                    'action': result['action'],
                    'positions_opened': result['positions_opened'],
                    'positions_closed': result['positions_closed']
                })
            
            # Calcular valor del portfolio
            portfolio_value = self.strategy.calculate_portfolio_value(row['Close'])
            equity_curve.append(portfolio_value)
            portfolio_values.append(portfolio_value)
            
            # Registrar estado de posiciones
            positions_history.append({
                'date': row['Date'],
                'open_positions': len(self.strategy.positions),
                'portfolio_value': portfolio_value,
                'available_capital': self.strategy.available_capital
            })
        
        # Cerrar posiciones restantes al final del período
        if len(self.strategy.positions) > 0:
            last_row = df_prepared.iloc[-1]
            for position in list(self.strategy.positions):
                self.strategy.close_position(
                    position, 
                    last_row['Date'], 
                    last_row['Close'], 
                    'end_of_period'
                )
        
        # Calcular métricas de performance
        performance_metrics = self.calculate_performance_metrics(
            equity_curve, 
            df_prepared, 
            self.strategy.trade_history
        )
        
        # Compilar resultados
        self.results = {
            'equity_curve': equity_curve,
            'portfolio_values': portfolio_values,
            'trades': self.strategy.trade_history,
            'trades_log': trades_log,
            'positions_history': positions_history,
            'performance_metrics': performance_metrics,
            'final_value': equity_curve[-1] if equity_curve else self.config['trading']['initial_capital'],
            'total_return': ((equity_curve[-1] / self.config['trading']['initial_capital']) - 1) * 100 if equity_curve else 0,
            'df_with_signals': df_prepared
        }
        
        if verbose:
            self.print_summary()
        
        return self.results
    
    def calculate_performance_metrics(self, equity_curve: List[float], 
                                     df: pd.DataFrame,
                                     trades: List[Dict]) -> Dict:
        """
        Calcula métricas detalladas de performance
        
        Parameters:
        -----------
        equity_curve : List[float]
            Curva de equity
        df : pd.DataFrame
            DataFrame con datos
        trades : List[Dict]
            Lista de trades ejecutados
            
        Returns:
        --------
        Dict
            Métricas de performance
        """
        if len(equity_curve) < 2:
            return self._empty_metrics()
        
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Filtrar returns válidos
        returns = returns[~np.isnan(returns)]
        
        # Calcular drawdown
        cumulative = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cumulative) / cumulative
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Métricas básicas
        total_return = ((equity_array[-1] / equity_array[0]) - 1) * 100
        
        # Convertir a retornos horarios y anualizarlos
        hours_per_year = 365 * 24
        
        # Sharpe Ratio
        if np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(hours_per_year)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(hours_per_year)
        else:
            sortino_ratio = 0
        
        # Calmar Ratio
        years = len(df) / hours_per_year
        annual_return = total_return / years if years > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown * 100) if max_drawdown != 0 else 0
        
        # Métricas de trades
        trades_metrics = self._calculate_trade_metrics(trades)
        
        metrics = {
            # Métricas de retorno
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return': annual_return / 12,
            
            # Ratios de riesgo-retorno
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Métricas de riesgo
            'max_drawdown': max_drawdown * 100,
            'volatility_annual': np.std(returns) * np.sqrt(hours_per_year) * 100 if len(returns) > 0 else 0,
            'var_95': np.percentile(returns, 5) * 100 if len(returns) > 20 else 0,
            
            # Métricas de trades
            **trades_metrics,
            
            # Información adicional
            'total_days': len(df) / 24,
            'total_years': years
        }
        
        return metrics
    
    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calcula métricas específicas de los trades
        
        Parameters:
        -----------
        trades : List[Dict]
            Lista de trades
            
        Returns:
        --------
        Dict
            Métricas de trades
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'avg_holding_hours': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        win_rate = (len(winning_trades) / len(trades_df)) * 100
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Expectancy (ganancia esperada por trade)
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
            'expectancy': expectancy,
            'avg_holding_hours': trades_df['holding_period'].mean() if 'holding_period' in trades_df.columns else 0,
            'max_consecutive_wins': self._max_consecutive(trades_df, 'win'),
            'max_consecutive_losses': self._max_consecutive(trades_df, 'loss')
        }
    
    def _max_consecutive(self, trades_df: pd.DataFrame, type: str) -> int:
        """
        Calcula el máximo de trades consecutivos ganadores o perdedores
        
        Parameters:
        -----------
        trades_df : pd.DataFrame
            DataFrame de trades
        type : str
            'win' o 'loss'
            
        Returns:
        --------
        int
            Máximo consecutivo
        """
        if len(trades_df) == 0:
            return 0
        
        if type == 'win':
            condition = trades_df['pnl'] > 0
        else:
            condition = trades_df['pnl'] <= 0
        
        consecutive = 0
        max_consecutive = 0
        
        for is_condition in condition:
            if is_condition:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _empty_metrics(self) -> Dict:
        """
        Retorna métricas vacías
        
        Returns:
        --------
        Dict
            Métricas vacías
        """
        return {
            'total_return': 0,
            'annual_return': 0,
            'monthly_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'volatility_annual': 0,
            'var_95': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0
        }
    
    def print_summary(self):
        """
        Imprime un resumen de los resultados del backtest
        """
        if not self.results:
            print("No hay resultados disponibles")
            return
        
        metrics = self.results['performance_metrics']
        
        print("\n" + "="*60)
        print("RESUMEN DEL BACKTEST")
        print("="*60)
        
        print("\n📊 RENDIMIENTO")
        print(f"Retorno Total: {metrics['total_return']:.2f}%")
        print(f"Retorno Anual: {metrics['annual_return']:.2f}%")
        print(f"Retorno Mensual Promedio: {metrics['monthly_return']:.2f}%")
        
        print("\n📈 RATIOS DE RIESGO-RETORNO")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        
        print("\n⚠️ MÉTRICAS DE RIESGO")
        print(f"Máximo Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Volatilidad Anual: {metrics['volatility_annual']:.2f}%")
        print(f"VaR 95%: {metrics['var_95']:.2f}%")
        
        print("\n💰 ESTADÍSTICAS DE TRADES")
        print(f"Total de Trades: {metrics['total_trades']}")
        print(f"Trades Ganadores: {metrics.get('winning_trades', 0)}")
        print(f"Trades Perdedores: {metrics.get('losing_trades', 0)}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Expectancy: ${metrics['expectancy']:.2f}")
        
        print("\n📅 INFORMACIÓN TEMPORAL")
        print(f"Período Total: {metrics.get('total_days', 0):.0f} días")
        print(f"Holding Period Promedio: {metrics.get('avg_holding_hours', 0):.1f} horas")
        
        print("\n💵 CAPITAL")
        print(f"Capital Inicial: ${self.config['trading']['initial_capital']:,.2f}")
        print(f"Capital Final: ${self.results['final_value']:,.2f}")
        print(f"P&L Total: ${self.results['final_value'] - self.config['trading']['initial_capital']:,.2f}")
        
        print("="*60)


# Función de prueba
def test_backtest_engine(config):
    """
    Prueba el motor de backtesting
    """
    # Crear instancia del motor
    engine = BacktestEngine(config)
    
    # Cargar datos
    loader = DataLoader(config)
    df = loader.load_data(config['data']['file_path'])
    train_data, test_data, val_data = loader.split_data()
    
    # Ejecutar backtest en datos de entrenamiento
    print("\n=== Backtest en Datos de Entrenamiento ===")
    results = engine.run_backtest(train_data, verbose=True)
    
    return engine, results


if __name__ == "__main__":
    from config import CONFIG
    engine, results = test_backtest_engine(CONFIG)