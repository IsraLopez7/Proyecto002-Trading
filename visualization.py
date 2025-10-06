"""
visualization.py
Módulo para generar reportes visuales completos del sistema de trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_comprehensive_report(data, portfolio_hist, trades, metrics_dict, best_params, save_path=None):
    """
    Crea un reporte visual completo estilo dashboard
    """
    # Crear figura principal con GridSpec
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('RESULTADOS DEL SISTEMA DE TRADING', fontsize=16, fontweight='bold', y=0.98)
    
    # Definir el grid
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25,
                          height_ratios=[1.2, 1, 1], width_ratios=[1.2, 1, 1, 0.8])
    
    # ========== Panel 1: Precio e Indicadores ==========
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Limitar datos para visualización
    display_data = data.tail(1000).reset_index(drop=True)
    x = range(len(display_data))
    
    # Graficar precio
    ax1.plot(x, display_data['close'], label='Precio', color='#2E86AB', linewidth=1.5)
    
    # Agregar medias móviles
    if f'sma_{best_params.get("sma_short", 10)}' in display_data.columns:
        ax1.plot(x, display_data[f'sma_{best_params.get("sma_short", 10)}'], 
                label='SMA Corta', color='orange', alpha=0.7, linewidth=1)
    if f'sma_{best_params.get("sma_long", 30)}' in display_data.columns:
        ax1.plot(x, display_data[f'sma_{best_params.get("sma_long", 30)}'], 
                label='SMA Larga', color='green', alpha=0.7, linewidth=1)
    if f'ema_{best_params.get("ema_period", 20)}' in display_data.columns:
        ax1.plot(x, display_data[f'ema_{best_params.get("ema_period", 20)}'], 
                label='EMA', color='red', alpha=0.5, linewidth=1, linestyle='--')
    
    ax1.set_title('BTC/USDT - Precio e Indicadores (últimos periodos)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Periodo')
    ax1.set_ylabel('Precio ($)')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========== Panel 2: Curva de Capital ==========
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Calcular retorno porcentual
    initial_capital = portfolio_hist[0] if isinstance(portfolio_hist, list) else portfolio_hist.iloc[0]
    final_capital = portfolio_hist[-1] if isinstance(portfolio_hist, list) else portfolio_hist.iloc[-1]
    total_return = (final_capital / initial_capital - 1) * 100
    
    # Graficar evolución del capital
    ax2.plot(range(len(portfolio_hist)), portfolio_hist, color='#2E86AB', linewidth=2)
    ax2.axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5, label='Capital Inicial')
    ax2.fill_between(range(len(portfolio_hist)), initial_capital, portfolio_hist, 
                     where=[p >= initial_capital for p in portfolio_hist],
                     color='green', alpha=0.1)
    ax2.fill_between(range(len(portfolio_hist)), initial_capital, portfolio_hist,
                     where=[p < initial_capital for p in portfolio_hist],
                     color='red', alpha=0.1)
    
    ax2.set_title(f'Curva de Capital (Retorno: {total_return:.2f}%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Periodo')
    ax2.set_ylabel('Capital ($)')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # ========== Panel 3: Drawdown ==========
    ax3 = fig.add_subplot(gs[1, :2])
    
    portfolio_series = pd.Series(portfolio_hist)
    cummax = portfolio_series.expanding().max()
    drawdown = (portfolio_series - cummax) / cummax * 100
    
    ax3.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    ax3.plot(drawdown, color='red', linewidth=1)
    
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    ax3.scatter(max_dd_idx, max_dd, color='darkred', s=50, zorder=5)
    
    ax3.set_title(f'Drawdown (Máximo: {max_dd:.2f}%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Periodo')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # ========== Panel 4: Retornos por Trade ==========
    ax4 = fig.add_subplot(gs[1, 2:])
    
    if trades and len(trades) > 0:
        df_trades = pd.DataFrame(trades)
        trade_returns = df_trades['return'].values * 100
        
        colors = ['green' if r > 0 else 'red' for r in trade_returns]
        ax4.bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.6)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        win_rate = (df_trades['pnl'] > 0).sum() / len(df_trades) * 100
        ax4.set_title(f'Retornos por Trade (Win Rate: {win_rate:.1f}%)', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No hay trades', ha='center', va='center', fontsize=12)
        ax4.set_title('Retornos por Trade', fontsize=12, fontweight='bold')
    
    ax4.set_xlabel('Trade #')
    ax4.set_ylabel('Retorno (%)')
    ax4.grid(True, alpha=0.3)
    
    # ========== Panel 5: Stochastic Oscillator ==========
    ax5 = fig.add_subplot(gs[2, :2])
    
    if f'stoch_k_{best_params.get("stoch_window", 14)}' in display_data.columns:
        stoch_k = display_data[f'stoch_k_{best_params.get("stoch_window", 14)}']
        stoch_d = display_data[f'stoch_d_{best_params.get("stoch_window", 14)}']
        
        ax5.plot(x, stoch_k, label='%K', color='blue', linewidth=1.5)
        ax5.plot(x, stoch_d, label='%D', color='orange', linewidth=1.5, alpha=0.7)
        
        # Zonas de sobrecompra y sobreventa
        ax5.axhline(y=best_params.get('stoch_overbought', 80), color='red', 
                   linestyle='--', alpha=0.5, label='Overbought')
        ax5.axhline(y=best_params.get('stoch_oversold', 20), color='green',
                   linestyle='--', alpha=0.5, label='Oversold')
        ax5.fill_between(x, best_params.get('stoch_oversold', 20), 
                        best_params.get('stoch_overbought', 80), alpha=0.05, color='gray')
        
        ax5.set_title('Stochastic', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 100)
        ax5.legend(loc='best', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'Stochastic no disponible', ha='center', va='center')
        ax5.set_title('Stochastic', fontsize=12, fontweight='bold')
    
    ax5.set_xlabel('Periodo')
    ax5.set_ylabel('Oscilador')
    ax5.grid(True, alpha=0.3)
    
    # ========== Panel 6: Tabla de Métricas ==========
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    
    # Crear tabla de métricas
    metrics_data = [
        ['Métrica', 'Valor'],
        ['Capital Final', f'${final_capital:,.2f}'],
        ['Retorno Total', f'{total_return:.2f}%'],
        ['CAGR', f'{metrics_dict.get("cagr", 0):.2f}%'],
        ['Sharpe', f'{metrics_dict.get("sharpe_ratio", 0):.3f}'],
        ['Sortino', f'{metrics_dict.get("sortino_ratio", 0):.3f}'],
        ['Calmar', f'{metrics_dict.get("calmar_ratio", 0):.3f}'],
        ['Max DD', f'{metrics_dict.get("max_drawdown", 0):.2f}%'],
        ['Trades', f'{metrics_dict.get("total_trades", 0):.0f}'],
        ['Win Rate', f'{metrics_dict.get("win_rate", 0):.1f}%'],
        ['PF', f'{metrics_dict.get("profit_factor", 0):.2f}']
    ]
    
    # Crear tabla
    table = ax6.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Estilo de la tabla
    for i in range(len(metrics_data)):
        if i == 0:  # Header
            table[(i, 0)].set_facecolor('#2E86AB')
            table[(i, 1)].set_facecolor('#2E86AB')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#f0f0f0')
    
    plt.tight_layout()
    
    # Guardar si se especifica
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def create_distribution_analysis(portfolio_hist, trades):
    """
    Crea análisis de distribución de retornos
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('ANÁLISIS DE DISTRIBUCIÓN DE RETORNOS', fontsize=14, fontweight='bold')
    
    # Calcular retornos
    portfolio_series = pd.Series(portfolio_hist)
    returns = portfolio_series.pct_change().dropna() * 100
    
    # 1. Histograma de retornos
    ax = axes[0, 0]
    ax.hist(returns, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Media: {returns.mean():.2f}%')
    ax.axvline(returns.median(), color='green', linestyle='--', label=f'Mediana: {returns.median():.2f}%')
    ax.set_title('Distribución de Retornos', fontsize=12, fontweight='bold')
    ax.set_xlabel('Retorno (%)')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    ax = axes[0, 1]
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normalidad)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Box Plot
    ax = axes[0, 2]
    box = ax.boxplot(returns, vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('#2E86AB')
    box['boxes'][0].set_alpha(0.7)
    ax.set_title('Box Plot de Retornos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retorno (%)')
    ax.grid(True, alpha=0.3)
    
    # 4. Retornos acumulados
    ax = axes[1, 0]
    cum_returns = (1 + returns/100).cumprod() - 1
    ax.plot(cum_returns.values * 100, color='#2E86AB', linewidth=2)
    ax.fill_between(range(len(cum_returns)), 0, cum_returns.values * 100, alpha=0.3, color='#2E86AB')
    ax.set_title('Retorno Acumulado', fontsize=12, fontweight='bold')
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Retorno Acumulado (%)')
    ax.grid(True, alpha=0.3)
    
    # 5. Rolling Sharpe Ratio
    ax = axes[1, 1]
    window = 252  # Ventana de 1 año para datos horarios
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(8760)
    ax.plot(rolling_sharpe, color='purple', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax.set_title(f'Rolling Sharpe Ratio ({window} periodos)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Periodo')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    
    # 6. Análisis de trades
    ax = axes[1, 2]
    if trades and len(trades) > 0:
        df_trades = pd.DataFrame(trades)
        
        # Separar ganancias y pérdidas
        wins = df_trades[df_trades['pnl'] > 0]['pnl'].values
        losses = df_trades[df_trades['pnl'] < 0]['pnl'].values
        
        # Crear histograma doble
        if len(wins) > 0:
            ax.hist(wins, bins=30, alpha=0.5, label=f'Ganancias (n={len(wins)})', color='green')
        if len(losses) > 0:
            ax.hist(losses, bins=30, alpha=0.5, label=f'Pérdidas (n={len(losses)})', color='red')
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Distribución de P&L por Trade', fontsize=12, fontweight='bold')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frecuencia')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No hay trades', ha='center', va='center')
        ax.set_title('Distribución de P&L por Trade', fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_monthly_performance_table(portfolio_hist, dates):
    try:
        import seaborn as sns
        _HAS_SEABORN = True
    except Exception:
        _HAS_SEABORN = False

    # DataFrame base
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'value': portfolio_hist
    }).set_index('date')

    # Retornos mensuales (%)
    monthly = df.resample('M').last().pct_change() * 100
    monthly['Year'] = monthly.index.year
    monthly['Month'] = monthly.index.month

    pivot_table = monthly.pivot_table(
        values='value', index='Year', columns='Month', aggfunc='first'
    ).sort_index()

    fig, ax = plt.subplots(figsize=(14, 8))

    if _HAS_SEABORN:
        month_names = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        sns.heatmap(
            pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Retorno (%)'},
            xticklabels=month_names, ax=ax, linewidths=0.5, linecolor='gray'
        )
        ax.set_title('Retornos Mensuales (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mes'); ax.set_ylabel('Año')
    else:
        ax.axis('off')
        ax.set_title('Retornos Mensuales (%) (sin seaborn)', fontsize=14, fontweight='bold')
        tbl = pivot_table.round(1).fillna('')
        table = ax.table(cellText=tbl.values, rowLabels=tbl.index, colLabels=tbl.columns,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)

    # ---- Columna "Anual" (fix: convertir a escalar float) ----
    annual = df.resample('Y').last().pct_change() * 100   # DataFrame con col "value"
    if 'value' in annual.columns:
        annual_series = annual['value']
    else:
        annual_series = annual.squeeze()

    # Escribe la columna "Anual" a la derecha del heatmap
    for i, year in enumerate(pivot_table.index):
        mask = (annual_series.index.year == year)
        if mask.any():
            annual_ret = float(annual_series.loc[mask].iloc[0])  # <-- aquí el cast a float
            ax.text(len(pivot_table.columns) + 0.5, i + 0.5, f'{annual_ret:.1f}%',
                    ha='center', va='center', fontweight='bold')

    ax.text(len(pivot_table.columns) + 0.5, -0.7, 'Anual', ha='center', fontweight='bold')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Test del módulo
    print("Módulo de generación de reportes cargado correctamente")
    print("Funciones disponibles:")
    print("  - create_comprehensive_report()")
    print("  - create_distribution_analysis()")
    print("  - create_monthly_performance_table()")