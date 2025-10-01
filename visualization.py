"""
Módulo de visualización
"""

import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self):
        pass
    
    def plot_results(self, df, equity, trades, metrics):
        """Crea visualizaciones de resultados"""
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Crear figura
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Precio con indicadores
        ax1 = plt.subplot(3, 2, 1)
        # Mostrar últimos 1000 períodos
        last_n = min(1000, len(df))
        ax1.plot(df['Close'].iloc[-last_n:], label='Precio', linewidth=1)
        if 'SMA_short' in df.columns:
            ax1.plot(df['SMA_short'].iloc[-last_n:], label='SMA Corto', alpha=0.7)
            ax1.plot(df['SMA_long'].iloc[-last_n:], label='SMA Largo', alpha=0.7)
        ax1.set_title('BTC/USDT - Últimos períodos')
        ax1.set_xlabel('Período')
        ax1.set_ylabel('Precio ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Curva de capital
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(equity, linewidth=2, color='green')
        ax2.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Capital Inicial')
        ax2.set_title(f'Curva de Capital (Retorno: {metrics["retorno_total"]:.2f}%)')
        ax2.set_xlabel('Período')
        ax2.set_ylabel('Capital ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = plt.subplot(3, 2, 3)
        equity_array = np.array(equity)
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax * 100
        ax3.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        ax3.plot(drawdown, color='red', linewidth=1)
        ax3.set_title(f'Drawdown (Máximo: {metrics["max_drawdown"]:.2f}%)')
        ax3.set_xlabel('Período')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribución de trades
        if trades:
            ax4 = plt.subplot(3, 2, 4)
            returns = [t['pnl_pct'] for t in trades]
            colors = ['green' if r > 0 else 'red' for r in returns]
            ax4.bar(range(len(returns)), returns, color=colors, alpha=0.6)
            ax4.set_title(f'Retornos por Trade (Win Rate: {metrics["win_rate"]:.1f}%)')
            ax4.set_xlabel('Trade #')
            ax4.set_ylabel('Retorno (%)')
            ax4.grid(True, alpha=0.3)
        
        # 5. RSI
        if 'RSI' in df.columns:
            ax5 = plt.subplot(3, 2, 5)
            last_n = min(500, len(df))
            ax5.plot(df['RSI'].iloc[-last_n:], color='purple')
            ax5.axhline(y=35, color='green', linestyle='--', alpha=0.5, label='Sobreventa')
            ax5.axhline(y=65, color='red', linestyle='--', alpha=0.5, label='Sobrecompra')
            ax5.set_title('RSI')
            ax5.set_xlabel('Período')
            ax5.set_ylabel('RSI')
            ax5.set_ylim([0, 100])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Tabla de métricas
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        # Crear tabla
        table_data = [
            ['Métrica', 'Valor'],
            ['Capital Final', f'${metrics["capital_final"]:,.0f}'],
            ['Retorno Total', f'{metrics["retorno_total"]:.2f}%'],
            ['Sharpe Ratio', f'{metrics["sharpe_ratio"]:.3f}'],
            ['Calmar Ratio', f'{metrics["calmar_ratio"]:.3f}'],
            ['Max Drawdown', f'{metrics["max_drawdown"]:.2f}%'],
            ['Total Trades', f'{metrics["total_trades"]}'],
            ['Win Rate', f'{metrics["win_rate"]:.1f}%']
        ]
        
        table = ax6.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        plt.suptitle('RESULTADOS DEL SISTEMA DE TRADING', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, results_dict):
        """Compara resultados entre conjuntos de datos"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sets = ['TRAIN', 'TEST', 'VALIDATION']
        colors = ['blue', 'orange', 'green']
        
        # Preparar datos
        metrics_names = ['retorno_total', 'sharpe_ratio', 'calmar_ratio']
        metrics_labels = ['Retorno Total (%)', 'Sharpe Ratio', 'Calmar Ratio']
        
        for i, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
            ax = axes[i]
            values = [results_dict[s]['metrics'][metric] for s in sets if s in results_dict]
            x = range(len(values))
            
            bars = ax.bar(x, values, color=colors[:len(values)])
            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels(sets[:len(values)])
            ax.grid(True, alpha=0.3)
            
            # Añadir valores en las barras
            for j, v in enumerate(values):
                ax.text(j, v, f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.suptitle('COMPARACIÓN DE MÉTRICAS POR CONJUNTO', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()