#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AJUSTE DE CURVAS - MÉTODO DOS MÍNIMOS QUADRADOS
Questão 2 - Item a) - Ajuste Linear (Reta)
Autor: Cálculo Numérico
Data: 2025-11-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================================
# PARTE 1: DEFINIÇÃO DOS DADOS
# ============================================================================

# Dados da tabela
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0.5, 0.6, 0.9, 0.8, 1.2, 1.5, 1.7, 2.0])

# Número de pontos
n = len(x)

print("="*75)
print("AJUSTE DE CURVAS - MÉTODO DOS MÍNIMOS QUADRADOS")
print("Ajuste Linear (Reta): y = ax + b")
print("="*75)

print("\n>>> DADOS:")
print(f"x = {x}")
print(f"y = {y}")
print(f"Número de pontos: n = {n}")


# ============================================================================
# PARTE 2: CÁLCULO DOS COEFICIENTES PELO MMQ
# ============================================================================

# Calcular somatórias necessárias
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_xy = np.sum(x * y)
mean_x = np.mean(x)
mean_y = np.mean(y)

print("\n>>> CÁLCULOS INTERMEDIÁRIOS (Somatórias):")
print(f"Σx    = {sum_x}")
print(f"Σy    = {sum_y:.1f}")
print(f"Σx²   = {sum_x2}")
print(f"Σxy   = {sum_xy:.1f}")
print(f"média(x) = {mean_x:.2f}")
print(f"média(y) = {mean_y:.2f}")

# Calcular coeficientes a (inclinação) e b (intercepto)
# Fórmulas do MMQ para ajuste linear:
# a = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
# b = (Σy - a*Σx) / n = média(y) - a*média(x)

numerador_a = n * sum_xy - sum_x * sum_y
denominador_a = n * sum_x2 - sum_x**2

a = numerador_a / denominador_a
b = (sum_y - a * sum_x) / n

print("\n>>> CÁLCULO DOS COEFICIENTES:")
print(f"\nCoeficiente angular (a):")
print(f"  a = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)")
print(f"  a = ({n}*{sum_xy:.1f} - {sum_x}*{sum_y:.1f}) / ({n}*{sum_x2} - {sum_x}²)")
print(f"  a = ({numerador_a:.1f}) / ({denominador_a})")
print(f"  a = {a:.6f}")

print(f"\nCoeficiente linear (b):")
print(f"  b = (Σy - a*Σx) / n")
print(f"  b = ({sum_y:.1f} - {a:.6f}*{sum_x}) / {n}")
print(f"  b = {b:.6f}")

print(f"\n>>> EQUAÇÃO DA RETA AJUSTADA:")
print(f"  y = {a:.4f}x + {b:.4f}")
print(f"  ou")
print(f"  y = {a:.6f}x + {b:.6f}")


# ============================================================================
# PARTE 3: CÁLCULO DE VALORES AJUSTADOS E RESÍDUOS
# ============================================================================

# Valores ajustados (preditos pelo modelo)
y_ajustado = a * x + b

# Resíduos (diferença entre valores observados e ajustados)
residuos = y - y_ajustado


# ============================================================================
# PARTE 4: CÁLCULO DE MÉTRICAS DE QUALIDADE
# ============================================================================

# Soma dos Quadrados Totais (SQT)
SQT = np.sum((y - mean_y)**2)

# Soma dos Quadrados dos Resíduos (SQR)
SQR = np.sum(residuos**2)

# Soma dos Quadrados Explicados (SQE)
SQE = np.sum((y_ajustado - mean_y)**2)

# Coeficiente de Determinação R²
R2 = 1 - (SQR / SQT)

# Alternativa: R² = SQE / SQT
R2_alt = SQE / SQT

# Erro Padrão
erro_padrao = np.sqrt(SQR / (n - 2))

# Correlação de Pearson
r_pearson = np.corrcoef(x, y)[0, 1]

# Erro Médio Absoluto
mae = np.mean(np.abs(residuos))

# Raiz do Erro Quadrático Médio
rmse = np.sqrt(np.mean(residuos**2))

print("\n>>> MÉTRICAS DE QUALIDADE DO AJUSTE:")
print(f"\nSoma dos Quadrados Totais (SQT):      {SQT:.6f}")
print(f"Soma dos Quadrados dos Resíduos (SQR): {SQR:.6f}")
print(f"Soma dos Quadrados Explicados (SQE):   {SQE:.6f}")
print(f"\nCoeficiente de Determinação (R²):     {R2:.6f}")
print(f"Correlação de Pearson (r):             {r_pearson:.6f}")
print(f"Erro Padrão:                           {erro_padrao:.6f}")
print(f"Erro Médio Absoluto (MAE):             {mae:.6f}")
print(f"Raiz do Erro Quadrático Médio (RMSE):  {rmse:.6f}")

print(f"\n>>> INTERPRETAÇÃO DO R²:")
if R2 >= 0.9:
    print(f"  R² = {R2:.4f} → Ajuste EXCELENTE (≥ 90%)")
elif R2 >= 0.7:
    print(f"  R² = {R2:.4f} → Ajuste BOM (70-90%)")
elif R2 >= 0.5:
    print(f"  R² = {R2:.4f} → Ajuste MODERADO (50-70%)")
else:
    print(f"  R² = {R2:.4f} → Ajuste FRACO (< 50%)")

print(f"  {R2*100:.2f}% da variabilidade dos dados é explicada pelo modelo linear.")


# ============================================================================
# PARTE 5: TABELA DE RESULTADOS DETALHADA
# ============================================================================

print("\n>>> TABELA DE RESULTADOS DETALHADA:")
print("-"*75)
print(f"{'i':<4} {'xi':<6} {'yi':<8} {'y_ajust':<12} {'resíduo':<12} {'resíduo²':<12}")
print("-"*75)

for i in range(n):
    print(f"{i+1:<4} {x[i]:<6} {y[i]:<8.2f} {y_ajustado[i]:<12.4f} "
          f"{residuos[i]:<12.4f} {residuos[i]**2:<12.6f}")

print("-"*75)
print(f"{'':>44} {'SOMA:':<12} {SQR:<12.6f}")
print("-"*75)


# ============================================================================
# PARTE 6: GERAÇÃO DE GRÁFICOS
# ============================================================================

print("\n>>> GERANDO GRÁFICOS...")

# GRÁFICO 1: Dispersão com reta ajustada
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Plotar dados observados
ax1.scatter(x, y, color='blue', s=120, label='Dados observados', 
           zorder=5, edgecolors='darkblue', linewidth=1.5, alpha=0.8)

# Plotar reta ajustada
x_linha = np.linspace(0, 9, 100)
y_linha = a * x_linha + b
ax1.plot(x_linha, y_linha, 'r-', linewidth=2.5, 
        label=f'Reta ajustada: y = {a:.4f}x + {b:.4f}', 
        zorder=3, alpha=0.8)

# Plotar linhas de resíduos
for i in range(n):
    ax1.plot([x[i], x[i]], [y[i], y_ajustado[i]], 'g--', 
            linewidth=1.5, alpha=0.5, zorder=2)

# Configurações do gráfico
ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('y', fontsize=13, fontweight='bold')
ax1.set_title('Ajuste Linear - Método dos Mínimos Quadrados', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlim(0, 9)
ax1.set_ylim(0, 2.5)

# Adicionar caixa de texto com R²
textstr = f'R² = {R2:.4f}\nr = {r_pearson:.4f}\nErro padrão = {erro_padrao:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('mmq_ajuste_linear.png', dpi=300, bbox_inches='tight')
print("  ✓ Gráfico 1 salvo: mmq_ajuste_linear.png")
plt.close()


# GRÁFICO 2: Análise de resíduos (4 subgráficos)
fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, figsize=(14, 10))

# Subgráfico 1: Resíduos vs x
ax2a.scatter(x, residuos, color='green', s=100, 
            edgecolors='darkgreen', linewidth=1.5, alpha=0.8)
ax2a.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2a.set_xlabel('x', fontsize=11, fontweight='bold')
ax2a.set_ylabel('Resíduos', fontsize=11, fontweight='bold')
ax2a.set_title('(a) Resíduos vs Variável Independente', 
              fontsize=12, fontweight='bold')
ax2a.grid(True, alpha=0.3)

# Subgráfico 2: Resíduos vs valores ajustados
ax2b.scatter(y_ajustado, residuos, color='purple', s=100, 
            edgecolors='darkviolet', linewidth=1.5, alpha=0.8)
ax2b.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2b.set_xlabel('Valores Ajustados', fontsize=11, fontweight='bold')
ax2b.set_ylabel('Resíduos', fontsize=11, fontweight='bold')
ax2b.set_title('(b) Resíduos vs Valores Ajustados', 
              fontsize=12, fontweight='bold')
ax2b.grid(True, alpha=0.3)

# Subgráfico 3: Histograma dos resíduos
ax2c.hist(residuos, bins=5, color='orange', edgecolor='darkorange', 
         alpha=0.7, linewidth=1.5)
ax2c.set_xlabel('Resíduos', fontsize=11, fontweight='bold')
ax2c.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax2c.set_title('(c) Histograma dos Resíduos', 
              fontsize=12, fontweight='bold')
ax2c.grid(True, alpha=0.3, axis='y')

# Subgráfico 4: Q-Q plot (normalidade dos resíduos)
stats.probplot(residuos, dist="norm", plot=ax2d)
ax2d.set_title('(d) Q-Q Plot dos Resíduos', 
              fontsize=12, fontweight='bold')
ax2d.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mmq_analise_residuos.png', dpi=300, bbox_inches='tight')
print("  ✓ Gráfico 2 salvo: mmq_analise_residuos.png")
plt.close()


# GRÁFICO 3: Comparação observados vs ajustados
fig3, ax3 = plt.subplots(figsize=(10, 6))

indices = np.arange(1, n+1)
width = 0.35

bars1 = ax3.bar(indices - width/2, y, width, label='Observados', 
               color='blue', edgecolor='darkblue', linewidth=1.5, alpha=0.7)
bars2 = ax3.bar(indices + width/2, y_ajustado, width, label='Ajustados',
               color='red', edgecolor='darkred', linewidth=1.5, alpha=0.7)

ax3.set_xlabel('Ponto i', fontsize=13, fontweight='bold')
ax3.set_ylabel('y', fontsize=13, fontweight='bold')
ax3.set_title('Comparação: Valores Observados vs Ajustados', 
             fontsize=14, fontweight='bold')
ax3.set_xticks(indices)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mmq_comparacao.png', dpi=300, bbox_inches='tight')
print("  ✓ Gráfico 3 salvo: mmq_comparacao.png")
plt.close()


# ============================================================================
# PARTE 7: RESUMO FINAL
# ============================================================================

print("\n" + "="*75)
print("✅ RESUMO FINAL")
print("="*75)
print(f"\n📊 Equação da reta ajustada:")
print(f"   y = {a:.6f}x + {b:.6f}")
print(f"\n📈 Qualidade do ajuste:")
print(f"   R² = {R2:.6f} ({R2*100:.2f}%)")
print(f"   Correlação (r) = {r_pearson:.6f}")
print(f"\n📉 Erro:")
print(f"   Erro Padrão = {erro_padrao:.6f}")
print(f"   RMSE = {rmse:.6f}")
print(f"   MAE = {mae:.6f}")
print(f"\n📁 Arquivos gerados:")
print(f"   • mmq_ajuste_linear.png")
print(f"   • mmq_analise_residuos.png")
print(f"   • mmq_comparacao.png")
print("\n" + "="*75)
print("✓ Programa finalizado com sucesso!")
print("="*75 + "\n")
