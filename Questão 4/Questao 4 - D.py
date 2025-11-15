import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Método de Euler Explícito
def euler_method(f, x0, y0, h, x_final):
    """Método de Euler Explícito"""
    n = int((x_final - x0) / h)
    x = np.linspace(x0, x0 + n*h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    return x, y

# Método de Runge-Kutta 2ª Ordem
def rk2_method(f, x0, y0, h, x_final):
    """Método de Runge-Kutta de 2ª Ordem"""
    n = int((x_final - x0) / h)
    x = np.linspace(x0, x0 + n*h, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + (h / 2) * (k1 + k2)
    return x, y

# Função diferencial
def f(x, y):
    """dy/dx = 0.04*y"""
    return 0.04 * y

# Solução analítica
def solucao_analitica(x):
    """y(x) = 1000*e^(0.04*x)"""
    return 1000 * np.exp(0.04 * x)

# Parâmetros
x0, y0 = 0, 1000
h_values = [1, 0.5, 0.25, 0.1]
x_finals = [1, 100, 1000]

print("="*120)
print("ANÁLISE DO PVI EM DIFERENTES INTERVALOS")
print("="*120)

# Dicionário para armazenar resultados
resultados = {}

# Análise para cada intervalo
for x_final in x_finals:
    y_true = solucao_analitica(x_final)
    
    print(f"\n{'─'*120}")
    print(f"INTERVALO: [0, {x_final}]")
    print(f"Solução Analítica em x={x_final}: {y_true:.10e}")
    print(f"Ordem de magnitude: 10^{np.log10(y_true):.2f}")
    print(f"{'─'*120}\n")
    
    dados_intervalo = []
    
    for h in h_values:
        # Executar Euler
        x_euler, y_euler = euler_method(f, x0, y0, h, x_final)
        
        # Executar RK2
        x_rk2, y_rk2 = rk2_method(f, x0, y0, h, x_final)
        
        # Calcular erros
        n_passos = len(x_euler) - 1
        y_euler_final = y_euler[-1]
        y_rk2_final = y_rk2[-1]
        
        erro_euler_abs = abs(y_euler_final - y_true)
        erro_rk2_abs = abs(y_rk2_final - y_true)
        
        erro_euler_rel = erro_euler_abs / y_true * 100 if y_true != 0 else 0
        erro_rk2_rel = erro_rk2_abs / y_true * 100 if y_true != 0 else 0
        
        razao = erro_euler_abs / erro_rk2_abs if erro_rk2_abs > 0 else np.inf
        
        dados_intervalo.append({
            'h': h,
            'Passos': n_passos,
            'y_Euler': y_euler_final,
            'Erro_Euler': erro_euler_abs,
            'y_RK2': y_rk2_final,
            'Erro_RK2': erro_rk2_abs,
            'Razão': razao
        })
    
    resultados[x_final] = dados_intervalo
    
    # Exibir tabela
    df = pd.DataFrame(dados_intervalo)
    print(df.to_string(index=False))

# Análise de convergência para cada intervalo
print(f"\n{'='*120}")
print("TAXA DE CONVERGÊNCIA (redução de erro ao reduzir h pela metade)")
print(f"{'='*120}\n")

for x_final in x_finals:
    print(f"Intervalo [0, {x_final}]:")
    dados = resultados[x_final]
    
    for i in range(len(dados)-1):
        h1, h2 = dados[i]['h'], dados[i+1]['h']
        taxa_euler = dados[i]['Erro_Euler'] / dados[i+1]['Erro_Euler'] if dados[i+1]['Erro_Euler'] > 0 else np.inf
        taxa_rk2 = dados[i]['Erro_RK2'] / dados[i+1]['Erro_RK2'] if dados[i+1]['Erro_RK2'] > 0 else np.inf
        print(f"  h = {h1} → h = {h2}: Euler = {taxa_euler:.4f}, RK2 = {taxa_rk2:.4f}")
    print()

# Gráficos comparativos (3x3)
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('Comportamento dos Métodos em Diferentes Intervalos', fontsize=16, fontweight='bold')

colors_euler = ['#1f77b4', '#3498db', '#5dade2', '#85c1e2']
colors_rk2 = ['#2ca02c', '#58d68d', '#73e6a0', '#a8edba']

for col_idx, x_final in enumerate(x_finals):
    y_true = solucao_analitica(x_final)
    x_analitica = np.linspace(x0, x_final, 100)
    y_analitica = solucao_analitica(x_analitica)
    
    # Gráfico 1: Euler vs Analítica
    ax = axes[col_idx, 0]
    for i, h in enumerate(h_values):
        x_e, y_e = euler_method(f, x0, y0, h, x_final)
        ax.plot(x_e, y_e, 'o-', label=f'h = {h}', color=colors_euler[i], markersize=2, alpha=0.7)
    ax.plot(x_analitica, y_analitica, 'k--', linewidth=2.5, label='Analítica')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y(x)', fontsize=10)
    ax.set_title(f'Euler - Intervalo [0, {x_final}]', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 2: RK2 vs Analítica
    ax = axes[col_idx, 1]
    for i, h in enumerate(h_values):
        x_r, y_r = rk2_method(f, x0, y0, h, x_final)
        ax.plot(x_r, y_r, 's-', label=f'h = {h}', color=colors_rk2[i], markersize=2, alpha=0.7)
    ax.plot(x_analitica, y_analitica, 'k--', linewidth=2.5, label='Analítica')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y(x)', fontsize=10)
    ax.set_title(f'RK2 - Intervalo [0, {x_final}]', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Gráfico 3: Convergência em escala log-log
    ax = axes[col_idx, 2]
    dados = resultados[x_final]
    h_vals = [d['h'] for d in dados]
    erros_e = [d['Erro_Euler'] for d in dados]
    erros_r = [d['Erro_RK2'] for d in dados]
    
    ax.loglog(h_vals, erros_e, 'bs-', linewidth=2, markersize=8, label='Euler')
    ax.loglog(h_vals, erros_r, 'go-', linewidth=2, markersize=8, label='RK2')
    ax.set_xlabel('h', fontsize=10)
    ax.set_ylabel(f'Erro em y({x_final})', fontsize=10)
    ax.set_title(f'Convergência - Intervalo [0, {x_final}]', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# Gráfico adicional: Comportamento em x_final=1000 com escala normal e logarítmica
print("\nGerando gráfico de evolução para o intervalo [0, 1000]...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_final = 1000
h_test = 0.1
x_e, y_e = euler_method(f, x0, y0, h_test, x_final)
x_r, y_r = rk2_method(f, x0, y0, h_test, x_final)
x_ana = np.linspace(x0, x_final, 100)
y_ana = solucao_analitica(x_ana)

# Escala normal
ax = axes[0]
ax.plot(x_e, y_e, 'b-', label=f'Euler h={h_test}', linewidth=1.5, alpha=0.7)
ax.plot(x_r, y_r, 'g-', label=f'RK2 h={h_test}', linewidth=1.5, alpha=0.7)
ax.plot(x_ana, y_ana, 'k--', label='Analítica', linewidth=2)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y(x)', fontsize=11)
ax.set_title(f'Evolução da Solução - Escala Normal [0, {x_final}]', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Escala logarítmica
ax = axes[1]
ax.semilogy(x_e, y_e, 'b-', label=f'Euler h={h_test}', linewidth=1.5, alpha=0.7)
ax.semilogy(x_r, y_r, 'g-', label=f'RK2 h={h_test}', linewidth=1.5, alpha=0.7)
ax.semilogy(x_ana, y_ana, 'k--', label='Analítica', linewidth=2)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y(x) - escala log', fontsize=11)
ax.set_title(f'Evolução da Solução - Escala Logarítmica [0, {x_final}]', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

print("\n" + "="*120)
print("ANÁLISE CONCLUÍDA")
print("="*120)