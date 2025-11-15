#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INTEGRAÇÃO NUMÉRICA - VERSÃO CORRIGIDA
Questão 3 - Item b)
Integral: ∫₀⁴ (3x² - 3x + 1) dx = 44
"""

import numpy as np

# ============================================================================
# FUNÇÃO CORRIGIDA
# ============================================================================

def f(x):
    """Função: f(x) = 3x² - 3x + 1"""
    return 3*x**2 - 3*x + 1

# ============================================================================
# REGRA DOS TRAPÉZIOS
# ============================================================================

def trapezios(f, a, b, n):
    """
    Calcula integral usando Regra dos Trapézios
    
    Fórmula: I ≈ (h/2)[f(x₀) + 2Σf(xᵢ) + f(xₙ)]
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    resultado = (y[0] + y[-1]) / 2.0
    resultado += np.sum(y[1:-1])
    resultado *= h
    return resultado

# ============================================================================
# REGRA DE SIMPSON
# ============================================================================

def simpson(f, a, b, n):
    """
    Calcula integral usando Regra de Simpson (1/3)
    
    Fórmula: I ≈ (h/3)[f(x₀) + 4Σf(x_ímpares) + 2Σf(x_pares) + f(xₙ)]
    NOTA: Simpson é EXATO para polinômios de grau ≤ 3
    """
    if n % 2 != 0:
        n += 1
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    resultado = y[0] + y[-1]
    resultado += 4.0 * np.sum(y[1:-1:2])    # Índices ímpares
    resultado += 2.0 * np.sum(y[2:-1:2])    # Índices pares
    resultado *= h / 3.0
    
    return resultado

# ============================================================================
# CÁLCULO
# ============================================================================

# Parâmetros
a = 0
b = 4
n = 4
h = 1.0

# Valor exato (analítico)
# ∫₀⁴ (3x² - 3x + 1) dx = [x³ - (3/2)x² + x]₀⁴ = 64 - 24 + 4 = 44
integral_exata = 44.0

# Pontos de avaliação
x_pontos = np.array([0, 1, 2, 3, 4])
y_pontos = f(x_pontos)

print("="*80)
print("INTEGRAÇÃO NUMÉRICA - QUESTÃO 3b (CORRIGIDA)")
print("="*80)

print("\n>>> INTEGRAL A CALCULAR:")
print("∫₀⁴ (3x² - 3x + 1) dx")

print("\n>>> PARÂMETROS:")
print(f"a = {a},  b = {b},  n = {n},  h = {h}")

print("\n>>> PONTOS E VALORES:")
print("-"*50)
print(f"{'i':<4} {'xᵢ':<8} {'f(xᵢ)':<15}")
print("-"*50)
for i, (xi, yi) in enumerate(zip(x_pontos, y_pontos)):
    print(f"{i:<4} {xi:<8.0f} {yi:<15.0f}")

# Calcular
integral_trap = trapezios(f, a, b, n)
integral_simp = simpson(f, a, b, n)

print("\n>>> CÁLCULO ANALÍTICO EXATO:")
print("∫₀⁴ (3x² - 3x + 1) dx")
print("= [x³ - (3/2)x² + x]₀⁴")
print("= [64 - 24 + 4] - [0]")
print(f"= {integral_exata:.1f}")

print("\n>>> RESULTADOS:")
print("-"*70)
print(f"Valor exato (analítico):     {integral_exata:.4f}")
print(f"Regra dos Trapézios (n=4):   {integral_trap:.4f}")
print(f"Regra de Simpson (n=4):      {integral_simp:.4f}")

# Calcular erros
erro_abs_trap = abs(integral_trap - integral_exata)
erro_abs_simp = abs(integral_simp - integral_exata)

erro_rel_trap = (erro_abs_trap / integral_exata) * 100 if integral_exata != 0 else 0
erro_rel_simp = (erro_abs_simp / integral_exata) * 100 if integral_exata != 0 else 0

print("\n>>> ERROS:")
print("-"*70)
print("\nTrapézios:")
print(f"  Erro absoluto:  {erro_abs_trap:.4f}")
print(f"  Erro relativo:  {erro_rel_trap:.4f}%")

print("\nSimpson:")
print(f"  Erro absoluto:  {erro_abs_simp:.4f}")
print(f"  Erro relativo:  {erro_rel_simp:.4f}%")

print("\n>>> CÁLCULO DO ERRO MÁXIMO TEÓRICO:")
print("-"*70)

print("\nTrapézios:")
print("  Fórmula: E ≤ (b-a)³/(12n²) × max|f''(x)|")
print("  f(x) = 3x² - 3x + 1")
print("  f'(x) = 6x - 3")
print("  f''(x) = 6  (constante)")
print(f"  E ≤ (4³)/(12×4²) × 6 = (64)/(192) × 6 = {(64/192)*6:.4f}")

print("\nSimpson:")
print("  Fórmula: E ≤ (b-a)⁵/(180n⁴) × max|f⁴(x)|")
print("  f(x) = 3x² - 3x + 1  (grau 2)")
print("  f⁴(x) = 0  (para polinômios de grau 2)")
print("  Simpson é EXATO para polinômios de grau ≤ 3")
print(f"  E = 0")

print("\n>>> TABELA RESUMIDA:")
print("-"*80)
print(f"{'Método':<20} {'Resultado':<15} {'Erro Abs':<15} {'Erro Rel':<15}")
print("-"*80)
print(f"{'Exato':<20} {integral_exata:<15.4f} {0:<15.6f} {0:<15.4f}%")
print(f"{'Trapézios':<20} {integral_trap:<15.4f} {erro_abs_trap:<15.6f} {erro_rel_trap:<15.4f}%")
print(f"{'Simpson':<20} {integral_simp:<15.4f} {erro_abs_simp:<15.6f} {erro_rel_simp:<15.4f}%")

print("\n>>> CONCLUSÕES:")
print("-"*80)
print(f"1. Valor exato: {integral_exata}")
print(f"\n2. Trapézios obtém {integral_trap} com erro de {erro_abs_trap} (aprox. 4.5%)")
print(f"   - Subestima o valor ao usar retas em vez de curvas")
print(f"\n3. Simpson obtém {integral_simp} com erro de {erro_abs_simp} (0.0%)")
print(f"   - Simpson é EXATO porque f(x) é polinômio de grau 2")
print(f"   - Simpson integra exatamente polinômios até grau 3")
print(f"\n4. Melhoria de Simpson: infinita (Simpson é exato)")
print(f"\n5. Para funções polinomiais de baixo grau:")
print(f"   - Sempre use Simpson quando possível")
print(f"   - Garante resultado com erro zero para grau ≤ 3")

print("\n" + "="*80)
print("✓ Programa finalizado com sucesso!")
print("="*80 + "\n")
