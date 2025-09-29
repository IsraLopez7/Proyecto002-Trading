# visualization.py
"""
Módulo de Visualización y Generación de Reportes
Crea gráficos profesionales y tablas de rendimiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PerformanceVisualizer:
    """
    Clase para crear visualizaciones de performance
    """
    
    def __init__(self, results: Dict, config: Dict):
        """
        Inicializa el visualizador
        
        Parameters:
        -----------
        results : Dict
            Resultados del backtest
        config : Dict
            Configuración del proyecto
        """
        self.results = results
        self.config = config
        self.figures = []
        
    def create_performance_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Crea un dashboard completo de performance
        
        Parameters:
        -----------
        save_path : str, optional
            Ruta para guardar la figura
            
        Returns:
        --------
        plt.Figure
            Figura del dashboard
        """
        # Crear figura grande con subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_drawdown(ax2)
        
        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_returns_distribution(ax3)
        
        # 4. Monthly Returns Heatmap
        ax4 = fig.add_subplot(gs[2, :2])
        self._plot_monthly_returns_heatmap(ax4)
        
        # 5. Métricas principales
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_metrics_table(ax5)
        
        # Título general
        fig.suptitle('Dashboard de Performance - Estrategia de Trading BTC', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def _plot_equity_curve(self, ax: plt.Axes):
        """
        Grafica la curva de equity
        """
        equity = self.results['equity_curve']
        df = self.results['df_with_signals']
        dates = df['Date'][:len(equity)]
        
        # Línea de equity
        ax.plot(dates, equity, label='Portfolio Value', linewidth=2, color='#2E86AB')
        
        # Línea de capital inicial
        ax.axhline(y=self.config['trading']['initial_capital'], 
                  color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Sombrear períodos de trades
        if 'trades' in self.results and self.results['trades']:
            for trade in self.results['trades']:
                if trade['position_type'] == 'LONG':
                    color = 'green'
                else:
                    color = 'red'
                
                ax.axvspan(trade['entry_date'], trade['exit_date'], 
                          alpha=0.1, color=color)
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Formato de fechas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Añadir estadísticas
        final_value = equity[-1]
        total_return = ((final_value / self.config['trading']['initial_capital']) - 1) * 100
        
        ax.text(0.02, 0.95, f'Final Value: ${final_value:,.0f}\nTotal Return: {total_return:.1f}%',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_drawdown(self, ax: plt.Axes):
        """
        Grafica el drawdown
        """
        equity = np.array(self.results['equity_curve'])
        df = self.results['df_with_signals']
        dates = df['Date'][:len(equity)]
        
        # Calcular drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        
        # Graficar
        ax.fill_between(dates, 0, drawdown, color='red', alpha=0.3)
        ax.plot(dates, drawdown, color='darkred', linewidth=1.5)
        
        # Marcar máximo drawdown
        max_dd_idx = np.argmin(drawdown)
        ax.plot(dates[max_dd_idx], drawdown[max_dd_idx], 'ro', markersize=8)
        ax.annotate(f'Max DD: {drawdown[max_dd_idx]:.1f}%',
                   xy=(dates[max_dd_idx], drawdown[max_dd_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([min(drawdown) * 1.1, 0])
        
        # Formato de fechas
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_returns_distribution(self, ax: plt.Axes):
        """
        Grafica la distribución de retornos
        """
        if 'trades' in self.results and self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            returns = trades_df['return_pct'].values
            
            # Histograma
            n, bins, patches = ax.hist(returns, bins=30, alpha=0.7, edgecolor='black')
            
            # Colorear barras según ganancia/pérdida
            for i, patch in enumerate(patches):
                if bins[i] >= 0:
                    patch.set_facecolor('green')
                else:
                    patch.set_facecolor('red')
            
            # Línea vertical en 0
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Estadísticas
            mean_return = np.mean(returns)
            ax.axvline(x=mean_return, color='blue', linestyle='-', linewidth=2, 
                      label=f'Mean: {mean_return:.2f}%')
            
            ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No trades executed', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    
    def _plot_monthly_returns_heatmap(self, ax: plt.Axes):
        """
        Crea un heatmap de retornos mensuales
        """
        equity = self.results['equity_curve']
        df = self.results['df_with_signals']
        
        # Crear DataFrame con equity y fechas
        equity_df = pd.DataFrame({
            'Date': df['Date'][:len(equity)],
            'Equity': equity
        })
        equity_df.set_index('Date', inplace=True)
        
        # Calcular retornos mensuales
        monthly_equity = equity_df.resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100
        
        # Crear matriz para heatmap
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        
        # Pivot table
        heatmap_data = monthly_returns.pivot_table(
            values='Equity', 
            index='Year', 
            columns='Month'
        )
        
        # Crear heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Nombres de meses
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)
    
    def _plot_metrics_table(self, ax: plt.Axes):
        """
        Crea una tabla con métricas principales
        """
        metrics = self.results['performance_metrics']
        
        # Preparar datos para la tabla
        table_data = [
            ['Total Return', f"{metrics['total_return']:.2f}%"],
            ['Annual Return', f"{metrics['annual_return']:.2f}%"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
            ['Sortino Ratio', f"{metrics['sortino_ratio']:.3f}"],
            ['Calmar Ratio', f"{metrics['calmar_ratio']:.3f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2f}%"],
            ['Win Rate', f"{metrics['win_rate']:.1f}%"],
            ['Total Trades', f"{metrics['total_trades']}"],
            ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
            ['Avg Win', f"${metrics.get('avg_win', 0):.0f}"],
            ['Avg Loss', f"${metrics.get('avg_loss', 0):.0f}"]
        ]
        
        # Crear tabla
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Colorear encabezados
        for i in range(2):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Colorear filas alternadas
        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#F0F0F0')
        
        ax.set_title('Key Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    
    def create_trade_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Crea análisis detallado de trades
        
        Parameters:
        -----------
        save_path : str, optional
            Ruta para guardar
            
        Returns:
        --------
        plt.Figure
            Figura con análisis de trades
        """
        if 'trades' not in self.results or not self.results['trades']:
            print("No hay trades para analizar")
            return None
        
        trades_df = pd.DataFrame(self.results['trades'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Trade Analysis', fontsize=16, fontweight='bold')
        
        # 1. P&L por trade
        ax = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in trades_df['pnl']]
        ax.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7)
        ax.set_title('P&L by Trade')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('P&L ($)')
        ax.grid(True, alpha=0.3)
        
        # 2. Holding period distribution
        ax = axes[0, 1]
        ax.hist(trades_df['holding_period'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title('Holding Period Distribution')
        ax.set_xlabel('Hours')
        ax.set_ylabel('Frequency')
        ax.axvline(trades_df['holding_period'].mean(), color='red', linestyle='--', 
                  label=f"Mean: {trades_df['holding_period'].mean():.1f}h")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Win/Loss by position type
        ax = axes[0, 2]
        position_types = trades_df['position_type'].unique()
        wins = []
        losses = []
        for ptype in position_types:
            type_trades = trades_df[trades_df['position_type'] == ptype]
            wins.append(len(type_trades[type_trades['pnl'] > 0]))
            losses.append(len(type_trades[type_trades['pnl'] <= 0]))
        
        x = np.arange(len(position_types))
        width = 0.35
        ax.bar(x - width/2, wins, width, label='Wins', color='green', alpha=0.7)
        ax.bar(x + width/2, losses, width, label='Losses', color='red', alpha=0.7)
        ax.set_title('Wins/Losses by Position Type')
        ax.set_xticks(x)
        ax.set_xticklabels(position_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Cumulative P&L
        ax = axes[1, 0]
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax.plot(cumulative_pnl, linewidth=2, color='#2E86AB')
        ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, alpha=0.3)
        ax.set_title('Cumulative P&L')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)
        
        # 5. Exit reasons
        ax = axes[1, 1]
        exit_reasons = trades_df['reason'].value_counts()
        colors_map = {'take_profit': 'green', 'stop_loss': 'red', 
                     'signal_reversal': 'orange', 'end_of_period': 'gray'}
        colors = [colors_map.get(r, 'blue') for r in exit_reasons.index]
        ax.pie(exit_reasons.values, labels=exit_reasons.index, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.set_title('Exit Reasons')
        
        # 6. P&L by signal strength
        ax = axes[1, 2]
        if 'signal_strength' in trades_df.columns:
            strength_bins = pd.cut(trades_df['signal_strength'], bins=3, 
                                  labels=['Weak', 'Medium', 'Strong'])
            avg_pnl_by_strength = trades_df.groupby(strength_bins)['pnl'].mean()
            
            colors = ['red' if x < 0 else 'green' for x in avg_pnl_by_strength]
            ax.bar(range(len(avg_pnl_by_strength)), avg_pnl_by_strength.values, 
                  color=colors, alpha=0.7)
            ax.set_xticks(range(len(avg_pnl_by_strength)))
            ax.set_xticklabels(avg_pnl_by_strength.index)
            ax.set_title('Average P&L by Signal Strength')
            ax.set_ylabel('Average P&L ($)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No signal strength data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Average P&L by Signal Strength')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def create_optimization_report(self, optimization_history: List[Dict],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Crea visualización del proceso de optimización
        
        Parameters:
        -----------
        optimization_history : List[Dict]
            Historial de optimización
        save_path : str, optional
            Ruta para guardar
            
        Returns:
        --------
        plt.Figure
            Figura con reporte de optimización
        """
        if not optimization_history:
            print("No hay historial de optimización")
            return None
        
        opt_df = pd.DataFrame(optimization_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Optimization Report', fontsize=16, fontweight='bold')
        
        # 1. Convergencia de optimización
        ax = axes[0, 0]
        ax.plot(opt_df['trial'], opt_df['calmar_ratio'], 'o-', alpha=0.6)
        ax.plot(opt_df['trial'], opt_df['calmar_ratio'].cummax(), 
               'r-', linewidth=2, label='Best so far')
        ax.set_title('Optimization Convergence')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Calmar Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Distribución de Calmar Ratios
        ax = axes[0, 1]
        ax.hist(opt_df['calmar_ratio'], bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(opt_df['calmar_ratio'].max(), color='red', linestyle='--',
                  label=f'Best: {opt_df["calmar_ratio"].max():.3f}')
        ax.set_title('Calmar Ratio Distribution')
        ax.set_xlabel('Calmar Ratio')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Top 10 trials
        ax = axes[1, 0]
        top_10 = opt_df.nlargest(10, 'calmar_ratio')
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10))
        ax.barh(range(10), top_10['calmar_ratio'].values, color=colors)
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"Trial {t}" for t in top_10['trial'].values])
        ax.set_title('Top 10 Trials')
        ax.set_xlabel('Calmar Ratio')
        ax.grid(True, alpha=0.3)
        
        # 4. Parameter importance (si hay suficientes datos)
        ax = axes[1, 1]
        if len(opt_df) > 20:
            # Calcular correlaciones de parámetros con Calmar
            params_cols = [col for col in opt_df.columns 
                          if col not in ['trial', 'calmar_ratio']]
            
            if params_cols:
                correlations = {}
                for col in params_cols:
                    if col in opt_df.columns and opt_df[col].dtype in ['float64', 'int64']:
                        corr = opt_df[col].corr(opt_df['calmar_ratio'])
                        correlations[col] = abs(corr)
                
                if correlations:
                    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:8]
                    params, values = zip(*sorted_corrs)
                    
                    colors = ['green' if v > 0.1 else 'orange' for v in values]
                    ax.barh(range(len(params)), values, color=colors, alpha=0.7)
                    ax.set_yticks(range(len(params)))
                    ax.set_yticklabels(params)
                    ax.set_title('Parameter Importance (|Correlation|)')
                    ax.set_xlabel('Absolute Correlation with Calmar Ratio')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No parameter data available', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No parameter columns found', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for analysis', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.figures.append(fig)
        return fig
    
    def generate_performance_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Genera tablas de rendimiento detalladas
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Diccionario con diferentes tablas de rendimiento
        """
        tables = {}
        
        # 1. Tabla de retornos por período
        equity = self.results['equity_curve']
        df = self.results['df_with_signals']
        
        equity_df = pd.DataFrame({
            'Date': df['Date'][:len(equity)],
            'Equity': equity
        })
        equity_df.set_index('Date', inplace=True)
        
        # Retornos mensuales
        monthly_returns = equity_df.resample('M').last().pct_change() * 100
        monthly_returns.columns = ['Return (%)']
        tables['monthly_returns'] = monthly_returns
        
        # Retornos trimestrales
        quarterly_returns = equity_df.resample('Q').last().pct_change() * 100
        quarterly_returns.columns = ['Return (%)']
        tables['quarterly_returns'] = quarterly_returns
        
        # Retornos anuales
        annual_returns = equity_df.resample('Y').last().pct_change() * 100
        annual_returns.columns = ['Return (%)']
        tables['annual_returns'] = annual_returns
        
        # 2. Tabla de métricas por año
        if 'trades' in self.results and self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df['Year'] = pd.to_datetime(trades_df['entry_date']).dt.year
            
            yearly_metrics = trades_df.groupby('Year').agg({
                'pnl': ['sum', 'mean', 'count'],
                'return_pct': 'mean'
            }).round(2)
            
            yearly_metrics.columns = ['Total P&L', 'Avg P&L', 'Num Trades', 'Avg Return (%)']
            tables['yearly_metrics'] = yearly_metrics
        
        # 3. Resumen estadístico
        summary_stats = pd.DataFrame([self.results['performance_metrics']]).T
        summary_stats.columns = ['Value']
        summary_stats = summary_stats.round(3)
        tables['summary_stats'] = summary_stats
        
        return tables
    
    def save_all_figures(self, base_path: str = 'reports/'):
        """
        Guarda todas las figuras generadas
        
        Parameters:
        -----------
        base_path : str
            Ruta base para guardar
        """
        import os
        
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        for i, fig in enumerate(self.figures):
            fig.savefig(f"{base_path}figure_{i+1}.png", dpi=300, bbox_inches='tight')
        
        print(f"Figuras guardadas en {base_path}")
    
    def generate_html_report(self, save_path: str = 'report.html'):
        """
        Genera un reporte HTML completo
        
        Parameters:
        -----------
        save_path : str
            Ruta para guardar el reporte HTML
        """
        tables = self.generate_performance_tables()
        metrics = self.results['performance_metrics']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #333; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #2E86AB; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric-box {{ 
                    display: inline-block; 
                    margin: 10px;
                    padding: 15px;
                    border: 2px solid #2E86AB;
                    border-radius: 5px;
                    min-width: 200px;
                }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Trading Strategy Performance Report - BTC/USDT</h1>
            
            <h2>Executive Summary</h2>
            <div>
                <div class="metric-box">
                    <strong>Total Return:</strong><br>
                    <span class="{'positive' if metrics['total_return'] > 0 else 'negative'}">
                        {metrics['total_return']:.2f}%
                    </span>
                </div>
                <div class="metric-box">
                    <strong>Calmar Ratio:</strong><br>
                    {metrics['calmar_ratio']:.3f}
                </div>
                <div class="metric-box">
                    <strong>Sharpe Ratio:</strong><br>
                    {metrics['sharpe_ratio']:.3f}
                </div>
                <div class="metric-box">
                    <strong>Max Drawdown:</strong><br>
                    <span class="negative">{metrics['max_drawdown']:.2f}%</span>
                </div>
                <div class="metric-box">
                    <strong>Win Rate:</strong><br>
                    {metrics['win_rate']:.1f}%
                </div>
                <div class="metric-box">
                    <strong>Total Trades:</strong><br>
                    {metrics['total_trades']}
                </div>
            </div>
            
            <h2>Performance Metrics</h2>
            {tables['summary_stats'].to_html()}
            
            <h2>Monthly Returns</h2>
            {tables['monthly_returns'].tail(12).to_html()}
            
            <h2>Annual Returns</h2>
            {tables['annual_returns'].to_html() if not tables['annual_returns'].empty else '<p>No annual data available</p>'}
            
            <p><i>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html)
        
        print(f"Reporte HTML guardado en {save_path}")


# Función de prueba
def test_visualization(results, config):
    """
    Prueba el módulo de visualización
    """
    visualizer = PerformanceVisualizer(results, config)
    
    # Crear dashboard principal
    dashboard = visualizer.create_performance_dashboard()
    
    # Crear análisis de trades
    trade_analysis = visualizer.create_trade_analysis()
    
    # Generar tablas
    tables = visualizer.generate_performance_tables()
    
    print("\n=== Visualizaciones Creadas ===")
    print("1. Dashboard de Performance")
    print("2. Análisis de Trades")
    print("3. Tablas de Rendimiento")
    
    return visualizer


if __name__ == "__main__":
    # Código de prueba
    pass