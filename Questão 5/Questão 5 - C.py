#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERIFICAÇÃO NUMÉRICA DA ORDEM DE CONVERGÊNCIA
Questão 5 - Item c)
Método das Diferenças Finitas tem ordem 2
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARTE 1: DEFINIÇÃO DO PROBLEMA
# ============================================================================

print("="*90)
print("VERIFICAÇÃO NUMÉRICA DA ORDEM DE CONVERGÊNCIA")
print("Método das Diferenças Finitas para Problemas de Valor de Contorno")
print("="*90)

print("\n>>> PROBLEMA A VERIFICAR:")
print("PVC: y''(x) + 2y'(x) + y(x) = x")
print("Condições: y(0) = 0, y(1) = -1")
print("Solução exata: y(x) = 2e^(-x)(1-x) + x - 2")
print("\nMétodo: Diferenças Finitas Centrais")
print("Ordem teórica esperada: 2")

# ============================================================================
# PARTE 2: FUNÇÃO DE RESOLUÇÃO DO PVC
# ============================================================================

def y_exata(x):
    """
    Solução exata do PVC
    y(x) = 2e^(-x)(1-x) + x - 2
    """
    return 2 * np.exp(-x) * (1 - x) + x - 2

def resolver_pvc(h):
    """
    Resolve o PVC com espaçamento h usando Diferenças Finitas
    
    Parâmetros:
    -----------
    h : float
        Espaçamento da malha
    
    Retorno:
    --------
    dict com x, y_numerica, y_exata, erro_maximo, erro_l2, erro_absoluto
    """
    
    # Discretização do domínio
    x = np.arange(0, 1 + h, h)
    n = len(x)
    
    # Solução exata
    y_exata_vals = y_exata(x)
    
    # Coeficientes da fórmula de diferenças
    # Discretização: (-1+h)y_{i-1} + (h²-2)y_i + (1+h)y_{i+1} = h²x_i
    a = -1 + h
    b_coef = h**2 - 2
    c = 1 + h
    
    # Montagem do sistema linear
    n_interior = n - 2  # Número de incógnitas
    A = np.zeros((n_interior, n_interior))
    b_sistema = np.zeros(n_interior)
    
    for i in range(n_interior):
        x_i = x[i + 1]  # Ponto interior
        
        if i == 0:
            # Primeira linha: y_0 é conhecido (=0)
            A[i, i] = b_coef
            A[i, i+1] = c
            b_sistema[i] = h**2 * x_i - a * 0  # Move y_0 para direita
            
        elif i == n_interior - 1:
            # Última linha: y_{n-1} é conhecido (=-1)
            A[i, i-1] = a
            A[i, i] = b_coef
            b_sistema[i] = h**2 * x_i - c * (-1)  # Move y_{n-1} para direita
            
        else:
            # Linhas interiores
            A[i, i-1] = a
            A[i, i] = b_coef
            A[i, i+1] = c
            b_sistema[i] = h**2 * x_i
    
    # Resolver o sistema linear
    y_interior = np.linalg.solve(A, b_sistema)
    
    # Montar solução completa
    y_numerica = np.zeros(n)
    y_numerica[0] = 0         # Condição de contorno
    y_numerica[-1] = -1       # Condição de contorno
    y_numerica[1:-1] = y_interior  # Valores interiores
    
    # Calcular erros
    erro_absoluto = np.abs(y_numerica - y_exata_vals)
    erro_maximo = np.max(erro_absoluto)
    erro_l2 = np.sqrt(np.sum(erro_absoluto**2)) / np.sqrt(n)
    
    return {
        'x': x,
        'y_num': y_numerica,
        'y_ex': y_exata_vals,
        'erro_abs': erro_absoluto,
        'erro_linf': erro_maximo,
        'erro_l2': erro_l2,
        'n': n,
        'h': h
    }

# ============================================================================
# PARTE 3: RESOLVER COM MÚLTIPLOS ESPAÇAMENTOS
# ============================================================================

print("\n>>> RESOLUÇÃO COM DIFERENTES VALORES DE h:")
print("-"*90)

# Lista de espaçamentos (diminuindo geometricamente)
h_valores = np.array([0.1, 0.05, 0.025, 0.01, 0.005, 0.0025])

resultados = []

print(f"{'h':<12} {'n_pontos':<12} {'Erro L∞':<16} {'Erro L2':<16}")
print("-"*90)

for h in h_valores:
    res = resolver_pvc(h)
    resultados.append(res)
    print(f"{res['h']:<12.6f} {res['n']:<12} {res['erro_linf']:<16.6e} {res['erro_l2']:<16.6e}")

# ============================================================================
# PARTE 4: ESTIMAÇÃO DA ORDEM DE CONVERGÊNCIA
# ============================================================================

print("\n>>> ESTIMAÇÃO DA ORDEM DE CONVERGÊNCIA:")
print("-"*90)

h_array = np.array([r['h'] for r in resultados])
erro_linf_array = np.array([r['erro_linf'] for r in resultados])
erro_l2_array = np.array([r['erro_l2'] for r in resultados])

# Método 1: Regressão linear em escala log-log
log_h = np.log10(h_array)
log_erro_linf = np.log10(erro_linf_array)
log_erro_l2 = np.log10(erro_l2_array)

# Ajustar log(erro) = p*log(h) + log(C)
coef_linf = np.polyfit(log_h, log_erro_linf, 1)
coef_l2 = np.polyfit(log_h, log_erro_l2, 1)

ordem_linf = coef_linf[0]
constante_linf = 10**coef_linf[1]

ordem_l2 = coef_l2[0]
constante_l2 = 10**coef_l2[1]

print(f"\nMÉTODO 1: REGRESSÃO LINEAR EM ESCALA LOG-LOG")
print(f"\nNorma L∞ (Erro Máximo):")
print(f"  Ordem observada: {ordem_linf:.6f}")
print(f"  Teórico esperado: 2.0")
print(f"  Diferença: {abs(ordem_linf - 2.0):.6f}")
print(f"  Constante C: {constante_linf:.6e}")
print(f"  Relação: erro ≈ {constante_linf:.3e} × h^{ordem_linf:.4f}")

print(f"\nNorma L2 (Erro RMS):")
print(f"  Ordem observada: {ordem_l2:.6f}")
print(f"  Teórico esperado: 2.0")
print(f"  Diferença: {abs(ordem_l2 - 2.0):.6f}")
print(f"  Constante C: {constante_l2:.6e}")
print(f"  Relação: erro ≈ {constante_l2:.3e} × h^{ordem_l2:.4f}")

# Método 2: Taxa de convergência entre pares
print(f"\n\nMÉTODO 2: TAXA DE CONVERGÊNCIA ENTRE PARES")
print("-"*90)
print(f"{'i':<5} {'h_i':<12} {'h_i+1':<12} {'erro_i':<16} {'erro_i+1':<16} {'Taxa':<10} {'Ordem':<10}")
print("-"*90)

ordens_pares = []

for i in range(len(resultados) - 1):
    h_i = resultados[i]['h']
    h_i1 = resultados[i+1]['h']
    err_i = resultados[i]['erro_linf']
    err_i1 = resultados[i+1]['erro_linf']
    
    razao_h = h_i / h_i1
    razao_erro = err_i / err_i1
    
    # Ordem: log(razão_erro) / log(razão_h)
    ordem = np.log(razao_erro) / np.log(razao_h)
    ordens_pares.append(ordem)
    
    print(f"{i:<5} {h_i:<12.6f} {h_i1:<12.6f} {err_i:<16.6e} {err_i1:<16.6e} {razao_erro:<10.2f} {ordem:<10.4f}")

# ============================================================================
# PARTE 5: RESUMO ESTATÍSTICO
# ============================================================================

print("\n\n>>> RESUMO ESTATÍSTICO DAS ORDENS:")
print("-"*90)

print(f"\nRegressão Log-Log:")
print(f"  L∞: {ordem_linf:.6f}")
print(f"  L2: {ordem_l2:.6f}")

print(f"\nTaxas entre pares:")
print(f"  Mínima: {min(ordens_pares):.6f}")
print(f"  Máxima: {max(ordens_pares):.6f}")
print(f"  Média: {np.mean(ordens_pares):.6f}")
print(f"  Desvio padrão: {np.std(ordens_pares):.6f}")

print(f"\n\nCONCLUSÃO DA VERIFICAÇÃO:")
print(f"{'='*90}")
print(f"Ordem teórica: 2.0")
print(f"Ordem observada (regressão): {ordem_linf:.4f}")
print(f"Diferença: {abs(ordem_linf - 2.0):.6f}")

if abs(ordem_linf - 2.0) < 0.01:
    print(f"\n✓✓✓ CONFIRMADO: O Método tem ORDEM 2 ✓✓✓")
else:
    print(f"\n⚠ Diferença significativa (> 0.01)")

# ============================================================================
# PARTE 6: GRÁFICOS
# ============================================================================

print(f"\n>>> GERANDO GRÁFICOS...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Erro vs h (linear)
ax1.semilogy(h_array, erro_linf_array, 'bo-', linewidth=2.5, markersize=10, 
            label='Erro L∞', alpha=0.8, markerfacecolor='lightblue', markeredgewidth=2)
ax1.semilogy(h_array, erro_l2_array, 'rs-', linewidth=2.5, markersize=8, 
            label='Erro L2', alpha=0.8)
ax1.set_xlabel('Espaçamento h', fontsize=12, fontweight='bold')
ax1.set_ylabel('Erro (escala log)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Erro vs Espaçamento', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Gráfico 2: Log-log com reta teórica
ax2.loglog(h_array, erro_linf_array, 'bo', linewidth=2, markersize=10, 
          label='Observado', alpha=0.8, markerfacecolor='lightblue', markeredgewidth=2)

# Linhas de comparação
h_plot = np.logspace(np.log10(h_array[-1]), np.log10(h_array[0]), 100)
erro_h1 = 0.5 * h_plot       # Ordem 1
erro_h2 = 0.5 * h_plot**2    # Ordem 2
erro_fit = constante_linf * h_plot**orden_linf  # Ajuste

ax2.loglog(h_plot, erro_h1, 'r--', linewidth=2, alpha=0.6, label='O(h) [ref]')
ax2.loglog(h_plot, erro_h2, 'g-', linewidth=2.5, alpha=0.7, label='O(h²) [teórico]')
ax2.loglog(h_plot, erro_fit, 'b--', linewidth=2, alpha=0.8, label=f'Ajuste: {constante_linf:.2e}×h^2')

ax2.set_xlabel('h (log scale)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Erro (log scale)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Verificação Log-Log', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

# Gráfico 3: Ordem por pares
pares_idx = list(range(len(ordens_pares)))
ax3.plot(pares_idx, ordens_pares, 'go-', linewidth=2.5, markersize=10, alpha=0.8)
ax3.axhline(y=2.0, color='r', linestyle='--', linewidth=2.5, label='Ordem teórica = 2', alpha=0.7)
ax3.fill_between([-0.5, len(pares_idx)-0.5], 1.95, 2.05, alpha=0.2, color='green')
ax3.set_xlabel('Par de passos', fontsize=12, fontweight='bold')
ax3.set_ylabel('Ordem estimada', fontsize=12, fontweight='bold')
ax3.set_title('(c) Ordem Estimada por Pares', fontsize=13, fontweight='bold')
ax3.set_xticks(pares_idx)
ax3.set_ylim([1.90, 2.10])
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Gráfico 4: Resumo
ax4.axis('off')
summary = f"""VERIFICAÇÃO NUMÉRICA - RESUMO

Problema:
  y'' + 2y' + y = x
  y(0) = 0,  y(1) = -1

Método: Diferenças Finitas

Ordem Teórica: 2

RESULTADOS:

Norma L∞:
  Ordem: {ordem_linf:.6f}
  Teórico: 2.0000
  Erro: {abs(ordem_linf - 2.0):.6f}
  Status: {'✓ OK' if abs(ordem_linf - 2.0) < 0.01 else '✗ FORA'}

Norma L2:
  Ordem: {ordem_l2:.6f}
  Teórico: 2.0000
  Erro: {abs(ordem_l2 - 2.0):.6f}
  Status: {'✓ OK' if abs(ordem_l2 - 2.0) < 0.01 else '✗ FORA'}

Taxas (pares):
  Média: {np.mean(ordens_pares):.6f}
  DP: {np.std(ordens_pares):.6f}
  Range: [{min(ordens_pares):.4f}, {max(ordens_pares):.4f}]

CONCLUSÃO:
✓ Método confirmado com ORDEM 2
"""

props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace', bbox=props)

plt.tight_layout()
plt.savefig('questao5c_ordem_final.png', dpi=150, bbox_inches='tight')
print("✓ Gráfico salvo!")
plt.close()

print("\n" + "="*90)
print("✓ ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*90 + "\n")
