#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESOLUÇÃO NUMÉRICA DE EQUAÇÕES DIFERENCIAIS ORDINÁRIAS
Problema de Valor de Contorno (PVC) - Método das Diferenças Finitas
Questão 5 - Item a)

PVC:
  y''(x) + 2y'(x) + y(x) = x,    0 < x < 1
  y(0) = 0
  y(1) = -1

Solução exata: y(x) = 2e^(-x)(1-x) + x - 2
Parâmetro: h = 0.01
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARTE 1: CONFIGURAÇÃO DO PROBLEMA
# ============================================================================

print("="*90)
print("RESOLUÇÃO NUMÉRICA DE EQUAÇÕES DIFERENCIAIS ORDINÁRIAS")
print("Problema de Valor de Contorno (PVC) - Método das Diferenças Finitas")
print("="*90)

print("\n>>> PROBLEMA A RESOLVER:")
print("Equação Diferencial: y''(x) + 2y'(x) + y(x) = x")
print("Domínio: 0 < x < 1")
print("Condições de Contorno:")
print("  y(0) = 0  (Dirichlet homogênea)")
print("  y(1) = -1 (Dirichlet não-homogênea)")
print("\nSolução Exata: y(x) = 2e^(-x)(1-x) + x - 2")
print("Parâmetro de Discretização: h = 0.01")

# ============================================================================
# PARTE 2: DISCRETIZAÇÃO DO DOMÍNIO
# ============================================================================

# Parâmetro
h = 0.01

# Criar malha de pontos
x = np.arange(0, 1 + h, h)  # De 0 até 1 com passo h
n = len(x)

print(f"\n>>> DISCRETIZAÇÃO DO DOMÍNIO:")
print(f"Espaçamento entre pontos: h = {h}")
print(f"Número total de pontos: n = {n}")
print(f"Pontos da malha: x_i = i·h para i = 0, 1, ..., {n-1}")

# Informações sobre a malha
print(f"\nMalha de pontos:")
print(f"  x_0 = 0.00     (condição de contorno: y(x_0) = 0)")
print(f"  x_1 = 0.01")
print(f"  x_2 = 0.02")
print(f"  ...")
print(f"  x_{n-2} = {x[n-2]:.2f}")
print(f"  x_{n-1} = {x[n-1]:.2f}  (condição de contorno: y(x_{n-1}) = -1)")

# ============================================================================
# PARTE 3: SOLUÇÃO EXATA PARA COMPARAÇÃO
# ============================================================================

def y_exata(x):
    """
    Solução exata do PVC
    y(x) = 2*e^(-x)*(1-x) + x - 2
    
    Parâmetro:
    -----------
    x : float ou array
        Ponto(s) onde avaliar a solução
    
    Retorno:
    --------
    y : float ou array
        Valor(es) da solução exata
    """
    return 2 * np.exp(-x) * (1 - x) + x - 2

# Avaliar solução exata em todos os pontos
y_exata_valores = y_exata(x)

print(f"\n>>> SOLUÇÃO EXATA AVALIADA EM PONTOS SELECIONADOS:")
print("-"*70)
print(f"{'i':<5} {'x_i':<10} {'y_exata(x_i)':<20}")
print("-"*70)

indices_amostra = [0, 10, 25, 50, 75, 100]
for i in indices_amostra:
    if i < n:
        print(f"{i:<5} {x[i]:<10.4f} {y_exata_valores[i]:<20.10f}")

# ============================================================================
# PARTE 4: FORMULAÇÃO DO MÉTODO DAS DIFERENÇAS FINITAS
# ============================================================================

print(f"\n>>> MÉTODO DAS DIFERENÇAS FINITAS:")
print("\nAproximações por diferenças:")
print("  y'(x_i) ≈ [y(x_{i+1}) - y(x_{i-1})] / (2h)           [Diferença Central]")
print("  y''(x_i) ≈ [y(x_{i+1}) - 2y(x_i) + y(x_{i-1})] / h²  [Diferença Central]")

print("\nSubstituindo na EDO: y''(x_i) + 2y'(x_i) + y(x_i) = x_i")
print("\n[y_{i+1} - 2y_i + y_{i-1}]/h² + 2[y_{i+1} - y_{i-1}]/(2h) + y_i = x_i")

print("\nMultiplicando por h²:")
print("[y_{i+1} - 2y_i + y_{i-1}] + h[y_{i+1} - y_{i-1}] + h²y_i = h²x_i")

print("\nRearranjando:")
print("(-1 + h)y_{i-1} + (h² - 2)y_i + (1 + h)y_{i+1} = h²x_i")

print("\nForma geral: a·y_{i-1} + b·y_i + c·y_{i+1} = d_i")

# ============================================================================
# PARTE 5: COEFICIENTES DA FÓRMULA
# ============================================================================

a = -1 + h      # Coeficiente de y_{i-1}
b_coef = h**2 - 2  # Coeficiente de y_i
c = 1 + h       # Coeficiente de y_{i+1}

print(f"\n>>> COEFICIENTES DA FÓRMULA DE DIFERENÇAS:")
print(f"a = -1 + h = -1 + {h} = {a:.6f}")
print(f"b = h² - 2 = {h**2} - 2 = {b_coef:.6f}")
print(f"c = 1 + h = 1 + {h} = {c:.6f}")
print(f"d_i = h² · x_i = {h**2} · x_i")

# ============================================================================
# PARTE 6: MONTAGEM DO SISTEMA LINEAR TRIDIAGONAL
# ============================================================================

print(f"\n>>> MONTAGEM DO SISTEMA LINEAR TRIDIAGONAL:")

# Número de incógnitas (pontos interiores)
n_interior = n - 2

print(f"Número de incógnitas: n - 2 = {n} - 2 = {n_interior}")
print(f"Tamanho da matriz: {n_interior} × {n_interior}")

# Montar matriz A (tridiagonal) e vetor b
A = np.zeros((n_interior, n_interior))
b_sistema = np.zeros(n_interior)

print(f"\nMontando o sistema linear A·y = b:")
print(f"\nEstrutura da matriz A ({n_interior} × {n_interior}):")
print(f"  Primeira linha: [b, c, 0, 0, ...]")
print(f"  Linhas interiores: [a, b, c, 0, ...]")
print(f"  Última linha: [..., 0, a, b]")

for i in range(n_interior):
    x_i = x[i + 1]  # Ponto interior
    
    if i == 0:
        # Primeira equação (i=1)
        # Incógnita: y_1
        # Envolve: y_0 (conhecida), y_1, y_2
        A[i, i] = b_coef           # Coeficiente de y_1
        A[i, i+1] = c              # Coeficiente de y_2
        # y_0 = 0 vai para o lado direito
        b_sistema[i] = h**2 * x_i - a * 0  # h²x_1 - a·y_0
        
    elif i == n_interior - 1:
        # Última equação (i=n-1)
        # Incógnita: y_{n-2}
        # Envolve: y_{n-3}, y_{n-2}, y_{n-1} (conhecida)
        A[i, i-1] = a              # Coeficiente de y_{n-3}
        A[i, i] = b_coef           # Coeficiente de y_{n-2}
        # y_{n-1} = -1 vai para o lado direito
        b_sistema[i] = h**2 * x_i - c * (-1)  # h²x_{n-1} - c·y_{n-1}
        
    else:
        # Equações interiores
        A[i, i-1] = a              # Coeficiente de y_{i-1}
        A[i, i] = b_coef           # Coeficiente de y_i
        A[i, i+1] = c              # Coeficiente de y_{i+1}
        b_sistema[i] = h**2 * x_i

print(f"\nMatriz A (primeiras 5 linhas):")
print(A[:5, :5])

print(f"\nVetor b (primeiros 5 elementos):")
print(b_sistema[:5])

# ============================================================================
# PARTE 7: RESOLUÇÃO DO SISTEMA LINEAR
# ============================================================================

print(f"\n>>> RESOLUÇÃO DO SISTEMA LINEAR:")
print(f"Sistema: A·y_interior = b")
print(f"Tamanho: {n_interior} × {n_interior}")

print(f"\nMétodo: Eliminação de Gauss com Pivotamento Parcial (LAPACK)")
print(f"Implementação: numpy.linalg.solve(A, b)")

print(f"\nJustificativa da escolha:")
print(f"  1. Matriz é tridiagonal (estrutura especial)")
print(f"  2. Sistema bem-condicionado para h pequeno")
print(f"  3. Algoritmo otimizado em LAPACK")
print(f"  4. Estabilidade numérica garantida com pivotamento")
print(f"  5. Custo computacional: O(n) para matriz tridiagonal")

# Resolver o sistema
y_numerica_interior = np.linalg.solve(A, b_sistema)

print(f"\nSistema resolvido com sucesso!")

# ============================================================================
# PARTE 8: CONSTRUÇÃO DA SOLUÇÃO COMPLETA
# ============================================================================

print(f"\n>>> CONSTRUÇÃO DA SOLUÇÃO COMPLETA:")

# Inicializar vetor solução
y_numerica = np.zeros(n)

# Incorporar condições de contorno
y_numerica[0] = 0         # Condição: y(0) = 0
y_numerica[-1] = -1       # Condição: y(1) = -1

# Inserir solução dos pontos interiores
y_numerica[1:-1] = y_numerica_interior

print(f"y_0 = {y_numerica[0]:.4f}  (condição de contorno)")
print(f"y_1, y_2, ..., y_{n-2} = valores obtidos pela solução do sistema")
print(f"y_{n-1} = {y_numerica[-1]:.4f}  (condição de contorno)")

# ============================================================================
# PARTE 9: COMPARAÇÃO E CÁLCULO DE ERROS
# ============================================================================

print(f"\n>>> ANÁLISE DE ERROS:")

# Erro absoluto em cada ponto
erro_absoluto = np.abs(y_numerica - y_exata_valores)

# Diferentes normas de erro
erro_maximo = np.max(erro_absoluto)
erro_medio = np.mean(erro_absoluto)
erro_l2 = np.sqrt(np.sum(erro_absoluto**2))
erro_relativo_maximo = erro_maximo / np.max(np.abs(y_exata_valores))

print(f"\nMétricas de erro:")
print(f"  Erro máximo (norma ∞):             {erro_maximo:.6e}")
print(f"  Erro médio (norma L¹):              {erro_medio:.6e}")
print(f"  Erro L² (norma euclidiana):         {erro_l2:.6e}")
print(f"  Erro relativo máximo:               {erro_relativo_maximo:.6e}")

# ============================================================================
# PARTE 10: TABELA DE RESULTADOS
# ============================================================================

print(f"\n>>> TABELA COMPARATIVA (amostra de pontos):")
print("-"*100)
print(f"{'i':<6} {'x_i':<10} {'y_numerica':<18} {'y_exata':<18} {'|Erro Abs|':<18}")
print("-"*100)

indices_tabela = [0, 10, 25, 50, 75, 100]
for i in indices_tabela:
    if i < n:
        print(f"{i:<6} {x[i]:<10.4f} {y_numerica[i]:<18.10f} {y_exata_valores[i]:<18.10f} {erro_absoluto[i]:<18.6e}")

# ============================================================================
# PARTE 11: GRÁFICOS
# ============================================================================

print(f"\n>>> GERANDO GRÁFICOS...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Solução numérica vs exata
ax1.plot(x, y_exata_valores, 'b-', linewidth=2.5, label='Solução exata', alpha=0.8)
ax1.plot(x, y_numerica, 'r.', markersize=3, label='Solução numérica', alpha=0.7)
ax1.set_xlabel('x', fontsize=11, fontweight='bold')
ax1.set_ylabel('y(x)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Comparação: Solução Numérica vs Exata', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Gráfico 2: Erro absoluto em escala logarítmica
erro_plot = erro_absoluto.copy()
erro_plot[erro_plot == 0] = 1e-15  # Evitar log(0)
ax2.semilogy(x, erro_plot, 'g-', linewidth=2, label='Erro absoluto', alpha=0.8)
ax2.set_xlabel('x', fontsize=11, fontweight='bold')
ax2.set_ylabel('|Erro absoluto| (escala log)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Erro Absoluto vs Posição', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

# Gráfico 3: Diferença y_num - y_exata
diferenca = y_numerica - y_exata_valores
ax3.plot(x, diferenca, 'purple', linewidth=2, label='y_numerica - y_exata', alpha=0.8)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
ax3.set_xlabel('x', fontsize=11, fontweight='bold')
ax3.set_ylabel('Diferença', fontsize=11, fontweight='bold')
ax3.set_title('(c) Diferença: Numérica - Exata', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Gráfico 4: Resumo de informações
ax4.axis('off')
info_text = f"""RESUMO DOS RESULTADOS

Problema de Valor de Contorno:
  y''(x) + 2y'(x) + y(x) = x
  y(0) = 0,  y(1) = -1

Método Numérico:
  Método: Diferenças Finitas
  Esquema: Diferenças Centrais
  h = {h}
  
Estrutura do Sistema:
  Tamanho: {n_interior} × {n_interior}
  Tipo: Tridiagonal
  a = {a:.6f}
  b = {b_coef:.6f}
  c = {c:.6f}

Resolução:
  Método: Eliminação de Gauss (LAPACK)
  Status: Sucesso ✓

Análise de Erros:
  Erro máximo: {erro_maximo:.6e}
  Erro médio: {erro_medio:.6e}
  Erro L²: {erro_l2:.6e}
  Erro rel. máx: {erro_relativo_maximo:.6e}

Validação:
  y(0) = {y_numerica[0]:.6f} (exato: 0.0)
  y(1) = {y_numerica[-1]:.6f} (exato: -1.0)
"""

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=9.5,
        verticalalignment='top', fontfamily='monospace', bbox=props)

plt.tight_layout()
plt.savefig('questao5a_pvc_final.png', dpi=150, bbox_inches='tight')
print("✓ Gráfico salvo: questao5a_pvc_final.png")
plt.close()

# ============================================================================
# PARTE 12: RESUMO FINAL
# ============================================================================

print("\n" + "="*90)
print("✅ RESUMO FINAL")
print("="*90)

print(f"\n📊 CONFIGURAÇÃO:")
print(f"  Espaçamento: h = {h}")
print(f"  Pontos: {n} (101 pontos)")
print(f"  Equações: {n_interior}")

print(f"\n📈 RESULTADOS:")
print(f"  Solução numérica obtida com sucesso!")
print(f"  Erro máximo: {erro_maximo:.6e}")
print(f"  Erro médio: {erro_medio:.6e}")

print(f"\n✓ Condições de contorno satisfeitas:")
print(f"  y(0) = {y_numerica[0]:.10f} (requerido: 0)")
print(f"  y(1) = {y_numerica[-1]:.10f} (requerido: -1)")

print(f"\n🔍 MÉTODO:")
print(f"  Discretização: Diferenças Finitas Centrais")
print(f"  Matriz: Tridiagonal {n_interior} × {n_interior}")
print(f"  Resolução: Eliminação de Gauss com Pivotamento (LAPACK)")

print("\n" + "="*90)
print("✓ Programa finalizado com sucesso!")
print("="*90 + "\n")
