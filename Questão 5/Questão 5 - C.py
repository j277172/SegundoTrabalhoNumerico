#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERIFICAÇÃO NUMÉRICA DA ORDEM DE CONVERGÊNCIA
Questão 5 - Item c)
Método das Diferenças Finitas tem ordem 2
"""

import numpy as np
import matplotlib.pyplot as plt

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

def y_exata(x):
    """Solução exata do PVC"""
    return 2 * np.exp(-x) * (1 - x) + x - 2

def resolver_pvc(h):
    """Resolve o PVC com espaçamento h usando Diferenças Finitas"""
    x = np.arange(0, 1 + h, h)
    n = len(x)
    y_exata_vals = y_exata(x)
    a = -1 + h
    b_coef = h**2 - 2
    c = 1 + h

    n_interior = n - 2  # Número de incógnitas
    A = np.zeros((n_interior, n_interior))
    b_sistema = np.zeros(n_interior)

    for i in range(n_interior):
        x_i = x[i + 1]  # Ponto interior
        if i == 0:
            A[i, i] = b_coef
            A[i, i+1] = c
            b_sistema[i] = h**2 * x_i - a * 0
        elif i == n_interior - 1:
            A[i, i-1] = a
            A[i, i] = b_coef
            b_sistema[i] = h**2 * x_i - c * (-1)
        else:
            A[i, i-1] = a
            A[i, i] = b_coef
            A[i, i+1] = c
            b_sistema[i] = h**2 * x_i

    y_interior = np.linalg.solve(A, b_sistema)
    y_numerica = np.zeros(n)
    y_numerica[0] = 0
    y_numerica[-1] = -1
    y_numerica[1:-1] = y_interior

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

print("\n>>> RESOLUÇÃO COM DIFERENTES VALORES DE h:")
print("-"*90)

h_valores = np.array([0.1, 0.05, 0.025, 0.01, 0.005, 0.0025])
resultados = []

print(f"{'h':<12} {'n_pontos':<12} {'Erro L∞':<16} {'Erro L2':<16}")
print("-"*90)

for h in h_valores:
    res = resolver_pvc(h)
    resultados.append(res)
    print(f"{res['h']:<12.6f} {res['n']:<12} {res['erro_linf']:<16.6e} {res['erro_l2']:<16.6e}")

print("\n>>> ESTIMAÇÃO DA ORDEM DE CONVERGÊNCIA:")
print("-"*90)
h_array = np.array([r['h'] for r in resultados])
erro_linf_array = np.array([r['erro_linf'] for r in resultados])
erro_l2_array = np.array([r['erro_l2'] for r in resultados])

log_h = np.log10(h_array)
log_erro_linf = np.log10(erro_linf_array)
log_erro_l2 = np.log10(erro_l2_array)

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

print("\n\nMÉTODO 2: TAXA DE CONVERGÊNCIA ENTRE PARES")
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
    ordem = np.log(razao_erro) / np.log(razao_h)
    ordens_pares.append(ordem)
    print(f"{i:<5} {h_i:<12.6f} {h_i1:<12.6f} {err_i:<16.6e} {err_i1:<16.6e} {razao_erro:<10.2f} {ordem:<10.4f}")

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

# (Gráficos removidos aqui para focar na parte numérica e essencial)
