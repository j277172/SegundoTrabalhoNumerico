#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AJUSTE DE CURVAS - MÉTODO DOS MÍNIMOS QUADRADOS
Questão 2 - Item b) - Ajuste Parabólico (Grau 2)
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

n = len(x)

print("="*80)
print("AJUSTE DE CURVAS - MÉTODO DOS MÍNIMOS QUADRADOS")
print("AJUSTE PARABÓLICO (Grau 2): y = ax² + bx + c")
print("="*80)

print("\n>>> DADOS:")
print(f"x = {x}")
print(f"y = {y}")
print(f"Número de pontos: n = {n}")


# ============================================================================
# PARTE 2: CÁLCULO DAS SOMATÓRIAS NECESSÁRIAS
# ============================================================================

# Somatórias para ajuste parabólico
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_x2 = np.sum(x**2)
sum_x3 = np.sum(x**3)
sum_x4 = np.sum(x**4)
sum_xy = np.sum(x * y)
sum_x2y = np.sum(x**2 * y)

print("\n>>> SOMATÓRIAS:")
print(f"Σx    = {sum_x}")
print(f"Σy    = {sum_y:.1f}")
print(f"Σx²   = {sum_x2}")
print(f"Σx³   = {sum_x3}")
print(f"Σx⁴   = {sum_x4}")
print(f"Σxy   = {sum_xy:.1f}")
print(f"Σx²y  = {sum_x2y:.1f}")


# ============================================================================
# PARTE 3: MONTAGEM E RESOLUÇÃO DO SISTEMA DE EQUAÇÕES NORMAIS
# ============================================================================

print("\n>>> SISTEMA DE EQUAÇÕES NORMAIS:")
print("Em forma matricial: A · coef = b")
print(f"\nMatriz A (3x3):")
print(f"[{sum_x4:>6}  {sum_x3:>6}  {sum_x2:>6}]   [a]     [{sum_x2y:>6.1f}]")
print(f"[{sum_x3:>6}  {sum_x2:>6}  {sum_x:>6}]   [b]  =  [{sum_xy:>6.1f}]")
print(f"[{sum_x2:>6}  {sum_x:>6}  {n:>6}]   [c]     [{sum_y:>6.1f}]")

# Montar matriz do sistema
A = np.array([
    [sum_x4, sum_x3, sum_x2],
    [sum_x3, sum_x2, sum_x],
    [sum_x2, sum_x, n]
])

b_vec = np.array([sum_x2y, sum_xy, sum_y])

# Resolver o sistema linear
coeficientes = np.linalg.solve(A, b_vec)
a_par = coeficientes[0]
b_par = coeficientes[1]
c_par = coeficientes[2]

print("\n>>> COEFICIENTES DA PARÁBOLA:")
print(f"a (coef. x²) = {a_par:.8f}")
print(f"b (coef. x)  = {b_par:.8f}")
print(f"c (constante) = {c_par:.8f}")

print(f"\n>>> EQUAÇÃO DA PARÁBOLA AJUSTADA:")
print(f"  y = {a_par:.6f}x² + {b_par:.6f}x + {c_par:.6f}")
print(f"  ou")
print(f"  y = {a_par:.4f}x² + {b_par:.4f}x + {c_par:.4f}")


# ============================================================================
# PARTE 4: CÁLCULO DE VALORES AJUSTADOS E RESÍDUOS
# ============================================================================

# Valores ajustados pela parábola
y_ajustado_par = a_par * x**2 + b_par * x + c_par

# Resíduos
residuos_par = y - y_ajustado_par


# ============================================================================
# PARTE 5: CÁLCULO DE MÉTRICAS DE QUALIDADE
# ============================================================================

# Soma dos Quadrados Totais
SQT = np.sum((y - np.mean(y))**2)

# Soma dos Quadrados dos Resíduos
SQR_par = np.sum(residuos_par**2)

# Coeficiente de Determinação R²
R2_par = 1 - (SQR_par / SQT)

# Erro Padrão
erro_padrao_par = np.sqrt(SQR_par / (n - 3))

# Correlação (entre valores observados e ajustados)
r_correlacao = np.corrcoef(y, y_ajustado_par)[0, 1]

print("\n>>> MÉTRICAS DE QUALIDADE DO AJUSTE:")
print(f"Soma dos Quadrados Totais (SQT):      {SQT:.6f}")
print(f"Soma dos Quadrados dos Resíduos (SQR): {SQR_par:.6f}")
print(f"Coeficiente de Determinação (R²):     {R2_par:.6f}")
print(f"Correlação (R² vs ajustado):          {r_correlacao:.6f}")
print(f"Erro Padrão:                          {erro_padrao_par:.6f}")

print(f"\n>>> INTERPRETAÇÃO DO R²:")
if R2_par >= 0.99:
    print(f"  R² = {R2_par:.4f} → Ajuste PRATICAMENTE PERFEITO (≥ 99%)")
elif R2_par >= 0.9:
    print(f"  R² = {R2_par:.4f} → Ajuste EXCELENTE (90-99%)")
elif R2_par >= 0.7:
    print(f"  R² = {R2_par:.4f} → Ajuste BOM (70-90%)")
else:
    print(f"  R² = {R2_par:.4f} → Ajuste MODERADO (< 70%)")

print(f"  {R2_par*100:.2f}% da variabilidade dos dados é explicada pelo modelo parabólico.")


# ============================================================================
# PARTE 6: COMPARAÇÃO COM AJUSTE LINEAR
# ============================================================================

# Ajuste linear (do item anterior)
a_reta = 0.228571
b_reta = 0.021429
y_ajustado_reta = a_reta * x + b_reta
SQR_reta = np.sum((y - y_ajustado_reta)**2)
R2_reta = 1 - (SQR_reta / SQT)

print("\n>>> COMPARAÇÃO: RETA vs PARÁBOLA")
print("-"*70)
print(f"{'Modelo':<15} {'Equação':<40} {'R²':<12}")
print("-"*70)
print(f"{'Reta':<15} {'y = 0.2286x + 0.0214':<40} {R2_reta:.6f}")
print(f"{'Parábola':<15} {f'y = {a_par:.4f}x² + {b_par:.4f}x + {c_par:.4f}':<40} {R2_par:.6f}")
print("-"*70)
print(f"Melhoria (ΔR²):  {R2_par - R2_reta:+.6f}  ({(R2_par - R2_reta)*100:+.2f}%)")
print(f"Redução SQR:     {(1 - SQR_par/SQR_reta)*100:.1f}%")


# ============================================================================
# PARTE 7: TABELA DE RESULTADOS DETALHADA
# ============================================================================

print("\n>>> TABELA DE RESULTADOS DETALHADA (PARÁBOLA):")
print("-"*85)
print(f"{'i':<4} {'xi':<6} {'yi':<8} {'y_ajust':<12} {'resíduo':<12} {'resíduo²':<12}")
print("-"*85)

for i in range(n):
    print(f"{i+1:<4} {x[i]:<6} {y[i]:<8.2f} {y_ajustado_par[i]:<12.4f} "
          f"{residuos_par[i]:<12.4f} {residuos_par[i]**2:<12.6f}")

print("-"*85)
print(f"{'':>44} {'SOMA:':<12} {SQR_par:<12.6f}")
print("-"*85)


# ============================================================================
# PARTE 8: GERAÇÃO DE GRÁFICOS
# ============================================================================

print("\n>>> GERANDO GRÁFICOS...")

# GRÁFICO 1: Comparação Reta vs Parábola
fig1, ax1 = plt.subplots(figsize=(12, 6))

# Dados
ax1.scatter(x, y, color='blue', s=120, label='Dados observados', 
           zorder=5, edgecolors='darkblue', linewidth=1.5, alpha=0.8)

# Reta
x_linha = np.linspace(0.5, 8.5, 200)
y_reta = a_reta * x_linha + b_reta
ax1.plot(x_linha, y_reta, 'r--', linewidth=2.5, 
        label=f'Reta: y = {a_reta:.4f}x + {b_reta:.4f} (R² = {R2_reta:.4f})',
        zorder=3, alpha=0.7)

# Parábola
y_par = a_par * x_linha**2 + b_par * x_linha + c_par
ax1.plot(x_linha, y_par, 'g-', linewidth=2.5, 
        label=f'Parábola: y = {a_par:.4f}x² + {b_par:.4f}x + {c_par:.4f} (R² = {R2_par:.4f})',
        zorder=3, alpha=0.8)

ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('y', fontsize=13, fontweight='bold')
ax1.set_title('Comparação: Ajuste Linear vs Ajuste Parabólico', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlim(0.5, 8.5)
ax1.set_ylim(0, 2.5)

plt.tight_layout()
plt.savefig('mmq_comparacao_reta_parabola.png', dpi=300, bbox_inches='tight')
print("  ✓ Gráfico 1 salvo: mmq_comparacao_reta_parabola.png")
plt.close()


# GRÁFICO 2: Parábola com Resíduos
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Dados
ax2.scatter(x, y, color='blue', s=120, label='Dados observados', 
           zorder=5, edgecolors='darkblue', linewidth=1.5, alpha=0.8)

# Parábola
ax2.plot(x_linha, y_par, 'g-', linewidth=2.5, label='Parábola ajustada',
        zorder=3, alpha=0.8)

# Linhas de resíduos
for i in range(n):
    ax2.plot([x[i], x[i]], [y[i], y_ajustado_par[i]], 'r--', 
            linewidth=1.5, alpha=0.5, zorder=2)

ax2.set_xlabel('x', fontsize=13, fontweight='bold')
ax2.set_ylabel('y', fontsize=13, fontweight='bold')
ax2.set_title('Ajuste Parabólico com Resíduos', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlim(0.5, 8.5)
ax2.set_ylim(0, 2.5)

# Info box
textstr = f'y = {a_par:.6f}x² + {b_par:.6f}x + {c_par:.6f}\nR² = {R2_par:.6f}'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('mmq_parabola_residuos.png', dpi=300, bbox_inches='tight')
print("  ✓ Gráfico 2 salvo: mmq_parabola_residuos.png")
plt.close()


# GRÁFICO 3: Análise de Resíduos (4 subgráficos)
fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, figsize=(14, 10))

# Subgráfico 1: Resíduos vs x
ax3a.scatter(x, residuos_par, color='green', s=100, 
            edgecolors='darkgreen', linewidth=1.5, alpha=0.8)
ax3a.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3a.set_xlabel('x', fontsize=11, fontweight='bold')
ax3a.set_ylabel('Resíduos', fontsize=11, fontweight='bold')
ax3a.set_title('(a) Resíduos vs x', fontsize=12, fontweight='bold')
ax3a.grid(True, alpha=0.3)

# Subgráfico 2: Resíduos vs valores ajustados
ax3b.scatter(y_ajustado_par, residuos_par, color='purple', s=100, 
            edgecolors='darkviolet', linewidth=1.5, alpha=0.8)
ax3b.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3b.set_xlabel('Valores Ajustados', fontsize=11, fontweight='bold')
ax3b.set_ylabel('Resíduos', fontsize=11, fontweight='bold')
ax3b.set_title('(b) Resíduos vs Valores Ajustados', fontsize=12, fontweight='bold')
ax3b.grid(True, alpha=0.3)

# Subgráfico 3: Histograma dos resíduos
ax3c.hist(residuos_par, bins=5, color='orange', edgecolor='darkorange', 
         alpha=0.7, linewidth=1.5)
ax3c.set_xlabel('Resíduos', fontsize=11, fontweight='bold')
ax3c.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax3c.set_title('(c) Histograma dos Resíduos', fontsize=12, fontweight='bold')
ax3c.grid(True, alpha=0.3, axis='y')

# Subgráfico 4: Q-Q plot
stats.probplot(residuos_par, dist="norm", plot=ax3d)
ax3d.set_title('(d) Q-Q Plot dos Resíduos', fontsize=12, fontweight='bold')
ax3d.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mmq_analise_residuos_parabola.png', dpi=300, bbox_inches='tight')
print("  ✓ Gráfico 3 salvo: mmq_analise_residuos_parabola.png")
plt.close()


# ============================================================================
# PARTE 9: RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("✅ RESUMO FINAL")
print("="*80)
print(f"\n📊 EQUAÇÃO DA PARÁBOLA:")
print(f"   y = {a_par:.8f}x² + {b_par:.8f}x + {c_par:.8f}")
print(f"\n📈 QUALIDADE DO AJUSTE:")
print(f"   R² (Parábola) = {R2_par:.6f}")
print(f"   R² (Reta)     = {R2_reta:.6f}")
print(f"   Melhoria      = {R2_par - R2_reta:+.6f}")
print(f"\n📉 ERRO:")
print(f"   Erro Padrão        = {erro_padrao_par:.6f}")
print(f"   SQR (Parábola)    = {SQR_par:.6f}")
print(f"   SQR (Reta)        = {SQR_reta:.6f}")
print(f"   Redução SQR       = {(1 - SQR_par/SQR_reta)*100:.1f}%")
print(f"\n📁 GRÁFICOS GERADOS:")
print(f"   • mmq_comparacao_reta_parabola.png")
print(f"   • mmq_parabola_residuos.png")
print(f"   • mmq_analise_residuos_parabola.png")
print("\n" + "="*80)
print("✓ Programa finalizado com sucesso!")
print("="*80 + "\n")
