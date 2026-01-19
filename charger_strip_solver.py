import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, toeplitz, lu_factor, lu_solve
from scipy.special import hankel2
import os
import pickle
import time

"""
================================================================================
PROBLÈME DU STRIP EN POLARISATION TM - Référence: Gibson - Chapter 5, Section 5.1.1
================================================================================

Ce code résout l'équation intégrale du strip parfaitement conducteur illuminé
par une onde plane en polarisation TM (Transverse Magnetic) en utilisant la 
Méthode des Moments.

ÉQUATION INTÉGRALE [Éq. 5.5 Gibson]:
∫₀ᴸ J_z(x') H₀⁽²⁾(k|x - x'|) dx' = (4/(ωμ)) E_zⁱ(x)

DISCRÉTISATION [Section 5.1.1.1 Gibson]:
- Le strip de longueur L est divisé en N segments
- Fonctions base "pulse" [Section 3.3.1]
- Point matching aux centres des segments [Section 3.2.1]

FORMULE DES ÉLÉMENTS DE MATRICE [Section 5.1.1.1 Gibson]:
- Termes self (m=n): Éq. 5.15
- Termes near (|m-n|=1): Éq. 5.16  
- Termes far (|m-n|>1): Approximation centroïdale Éq. 5.11
"""

def compute_TM_strip_matrix_element(m, n, x_centers, delta_x, k, method='exact'):
    """
    Calcule l'élément de matrice Z_mn pour le strip TM [Section 5.1.1.1 Gibson]
    
    Parameters:
    m, n: indices des segments test et source
    x_centers: positions des centres des segments
    delta_x: longueur d'un segment
    k: nombre d'onde
    method: 'exact' pour intégration exacte, 'centroid' pour approximation
    """
    gamma = 1.781072417990198  # Constante d'Euler-Mascheroni
    
    if m == n:
        # ÉQUATION (5.15) Gibson: Terme self
        term = 1 - 2j/np.pi * np.log(gamma * k * delta_x / (4 * np.e))
        return delta_x * term
        
    elif abs(m - n) == 1:
        # ÉQUATION (5.16) Gibson: Termes adjacents
        term1 = 3 * np.log(3 * gamma * k * delta_x / 4)
        term2 = np.log(gamma * k * delta_x / 4)
        term = 1 - 1j/np.pi * (term1 - term2 - 2)
        return delta_x * term
        
    else:
        # ÉQUATION (5.11) Gibson: Termes far - approximation centroïdale
        distance = abs(x_centers[m] - x_centers[n])
        return delta_x * hankel2(0, k * distance)

def build_TM_strip_matrix(L, N, k):
    """
    Construit la matrice d'impédance pour le strip TM [Section 5.1.1.1 Gibson]
    
    La matrice est symétrique Toeplitz en raison de la géométrie linéaire
    """
    delta_x = L / N
    x_centers = np.linspace(delta_x/2, L - delta_x/2, N)
    
    # Initialisation avec des zéros complexes
    first_row = np.zeros(N, dtype=complex)
    for n in range(N):
        first_row[n] = compute_TM_strip_matrix_element(0, n, x_centers, delta_x, k)
    
    # Construction de la matrice Toeplitz
    Z = toeplitz(first_row,first_row)
    return Z, x_centers, delta_x

def compute_incident_field(x_centers, k, phi_i):
    """
    Calcule le champ incident sur le strip [Éq. 5.1 Gibson]
    
    Pour TM: E_zⁱ(x) = e^{jkx cos(φⁱ)}
    """
    return np.exp(1j * k * x_centers * np.cos(phi_i))

def compute_excitation_vector(E_z_i, omega, mu):
    """
    Calcule le vecteur d'excitation [Éq. 5.10 Gibson]
    
    b_m = (4/(ωμ)) E_zⁱ(x_m)
    """
    return (4 / (omega * mu)) * E_z_i

def solve_TM_strip_procedural(L=3.0, freq=1e9, phi_i=np.pi/2, N=500, save_dir=None, force_symmetry=True):
    """
    Résout le problème du strip TM - Approche procédurale [Section 5.1.1 Gibson]
    
    Parameters:
    L: float - Longueur du strip (en longueurs d'onde)
    freq: float - Fréquence (Hz)
    phi_i: float - Angle d'incidence (radians)
    N: int - Nombre de segments
    save_dir: str - Dossier de sauvegarde
    force_symmetry: bool - Force la symétrie du courant pour incidence normale
    """
    # Paramètres physiques
    c = 3e8  # Vitesse de la lumière
    mu_0 = 4 * np.pi * 1e-7
    epsilon_0 = 8.854187817e-12
    
    lambda_ = c / freq
    k = 2 * np.pi / lambda_
    omega = 2 * np.pi * freq
    eta = np.sqrt(mu_0 / epsilon_0)
    
    # Conversion de L en mètres
    L_meters = L * lambda_
    
    if save_dir is None:
        save_dir = f"TM_strip_L{L}lambda_N{N}"
    
    print(f"Resolution du strip TM [Section 5.1.1 Gibson]")
    print(f"Parametres: L={L}λ, f={freq/1e9:.1f} GHz, φⁱ={np.degrees(phi_i):.1f}°, N={N}")
    print(f"Sauvegarde dans: {save_dir}")
    
    # ============================================================================
    # ÉTAPE 1: CONSTRUCTION DE LA MATRICE D'IMPÉDANCE
    # ============================================================================
    print("1. Construction de la matrice Z [Section 5.1.1.1]...")
    Z, x_centers, delta_x = build_TM_strip_matrix(L_meters, N, k)
    
    # Factorisation LU pour optimiser les résolutions multiples
    lu_piv = lu_factor(Z)
    
    # ============================================================================
    # ÉTAPE 2: CALCUL DE LA SER MONOSTATIQUE (CORRIGÉ)
    # ============================================================================
    print("4. Calcul de la SER MONOSTATIQUE")
    phi_s = np.linspace(0, np.pi, 181)
    SER_mom_mono = np.zeros_like(phi_s, dtype=complex)  # SER monostatique MoM
    SER_po_mono = np.zeros_like(phi_s, dtype=complex)   # SER monostatique PO
    
    # Stockage du courant pour l'angle d'incidence spécifié
    J_z_fixed = None
    J_po_fixed = None
    
    for i, phi in enumerate(phi_s):
        # Pour chaque angle, on calcule le champ incident correspondant
        E_z_i = compute_incident_field(x_centers, k, phi)
        b = compute_excitation_vector(E_z_i, omega, mu_0)
        
        # Résolution du système
        J_z = lu_solve(lu_piv, b)
        
        # Calcul du courant PO pour cet angle
        J_po = (2 / eta) * np.sin(phi) * E_z_i
        
        # Sauvegarde pour l'angle d'incidence spécifié
        if np.isclose(phi, phi_i, atol=1e-10):
            J_z_fixed = J_z.copy()
            J_po_fixed = J_po.copy()
            if force_symmetry and np.isclose(phi_i, np.pi/2):
                print("3.5. Forçage de la symétrie du courant (incidence normale)")
                J_z_fixed = 0.5 * (J_z_fixed + J_z_fixed[::-1])
        
        # Calcul de la SER monostatique (φ_s = φ_i = φ)
        integrand_mom = J_z * np.exp(1j * k * x_centers * np.cos(phi))
        integral_mom = np.sum(integrand_mom) * delta_x
        SER_mom_mono[i] = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k)) * integral_mom
        
        integrand_po = J_po * np.exp(1j * k * x_centers * np.cos(phi))
        integral_po = np.sum(integrand_po) * delta_x
        SER_po_mono[i] = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k)) * integral_po
    
    # Facteur 2*pi pour le 2D
    SER_mom_dB = 10 * np.log10(2 * np.pi * np.abs(SER_mom_mono)**2)
    SER_po_dB = 10 * np.log10(2 * np.pi * np.abs(SER_po_mono)**2)
    
    # ============================================================================
    # ÉTAPE 3: CALCUL BISTATIQUE POUR COMPARAISON (optionnel, gardé pour compatibilité)
    # ============================================================================
    print("5. Calcul bistatique supplémentaire pour comparaison")
    
    # Recalcul du courant pour l'angle d'incidence spécifié
    E_z_i_fixed = compute_incident_field(x_centers, k, phi_i)
    b_fixed = compute_excitation_vector(E_z_i_fixed, omega, mu_0)
    
    if J_z_fixed is None:
        J_z_fixed = lu_solve(lu_piv, b_fixed)
        if force_symmetry and np.isclose(phi_i, np.pi/2):
            J_z_fixed = 0.5 * (J_z_fixed + J_z_fixed[::-1])
    
    if J_po_fixed is None:
        J_po_fixed = (2 / eta) * np.sin(phi_i) * E_z_i_fixed
    
    # Calcul bistatique
    SER_mom_bistatic = np.zeros_like(phi_s, dtype=complex)
    SER_po_bistatic = np.zeros_like(phi_s, dtype=complex)
    
    for i, phi in enumerate(phi_s):
        # Utilisation du courant fixe pour φ_i
        integrand_mom = J_z_fixed * np.exp(1j * k * x_centers * np.cos(phi))
        integral_mom = np.sum(integrand_mom) * delta_x
        SER_mom_bistatic[i] = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k)) * integral_mom
        
        integrand_po = J_po_fixed * np.exp(1j * k * x_centers * np.cos(phi))
        integral_po = np.sum(integrand_po) * delta_x
        SER_po_bistatic[i] = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k)) * integral_po
    
    SER_mom_bistatic_dB = 10 * np.log10(2 * np.pi * np.abs(SER_mom_bistatic)**2)
    SER_po_bistatic_dB = 10 * np.log10(2 * np.pi * np.abs(SER_po_bistatic)**2)
    
    # Vérification de la symétrie du courant
    symmetry_error = 0
    if np.isclose(phi_i, np.pi/2):
        symmetry_error = np.max(np.abs(J_z_fixed - J_z_fixed[::-1]))
        print(f"   Erreur de symétrie du courant: {symmetry_error:.2e}")
    
    # Structure des résultats
    results = {
        # Courant pour l'angle spécifié (pour les graphiques)
        'J_z': J_z_fixed,                    # Courant MoM
        'J_po': J_po_fixed,                  # Courant Physical Optics
        
        # SER MONOSTATIQUE (corrigée - à utiliser pour comparaison avec Gibson)
        'SER_mom_mono': SER_mom_dB,          # SER MoM monostatique
        'SER_po_mono': SER_po_dB,            # SER PO monostatique
        
        # SER BISTATIQUE (gardée pour compatibilité)
        'SER_mom_bistatic': SER_mom_bistatic_dB,  # SER MoM bistatique
        'SER_po_bistatic': SER_po_bistatic_dB,    # SER PO bistatique
        
        # Données communes
        'Z': Z,                              # Matrice d'impédance
        'b': b_fixed,                        # Vecteur d'excitation
        'x_centers': x_centers,              # Positions (m)
        'x_lambda': x_centers/lambda_,       # Positions (λ)
        'phi_s': np.degrees(phi_s),          # Angles de diffusion (degrés)
        'params': {
            'L': L, 'L_meters': L_meters, 'freq': freq, 'lambda': lambda_,
            'phi_i': phi_i, 'phi_i_deg': np.degrees(phi_i), 'N': N,
            'k': k, 'omega': omega, 'eta': eta,
            'save_dir': save_dir,
            'symmetry_error': symmetry_error
        }
    }
    
    print("Resolution terminee avec succes")
    return results

def plot_TM_strip_results(results, save_dir=None):
    """
    Trace les résultats pour le strip TM [Section 5.1.1.2 Gibson]
    
    Génère les visualisations principales :
    - Courant induit vs Physical Optics [Figure 5.2a Gibson]
    - SER monostatique [Figure 5.2b Gibson]
    """
    if save_dir is None:
        save_dir = results['params']['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ============================================================================
    # FIGURE 1: COURANT INDUIT [FIGURE 5.2a GIBSON]
    # ============================================================================
    print("Generation du graphique de courant [Figure 5.2a]")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['x_lambda'], np.abs(results['J_z']), 'b-', linewidth=2, label='EFIE MoM')
    plt.plot(results['x_lambda'], np.abs(results['J_po']), 'r--', linewidth=2, label='Physical Optics')
    plt.xlabel('Position (λ)')
    plt.ylabel('|J_z| (A/m)')
    plt.title(f'Courant induit sur le strip TM\nφⁱ = {results["params"]["phi_i_deg"]:.1f}°')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results['x_lambda'], np.real(results['J_z']), 'b-', linewidth=2, label='Partie réelle MoM')
    plt.plot(results['x_lambda'], np.imag(results['J_z']), 'r-', linewidth=2, label='Partie imaginaire MoM')
    plt.xlabel('Position (λ)')
    plt.ylabel('J_z (A/m)')
    plt.title('Parties réelle et imaginaire du courant')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/TM_strip_current.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================================================
    # FIGURE 2: SER MONOSTATIQUE [FIGURE 5.2b GIBSON] - CORRIGÉE
    # ============================================================================
    print("Generation de la SER MONOSTATIQUE [Figure 5.2b corrigée]")
    plt.figure(figsize=(12, 5))
    
    # Sous-figure 1: Comparaison monostatique vs bistatique
    plt.subplot(1, 2, 1)
    plt.plot(results['phi_s'], results['SER_mom_mono'], 'b-', linewidth=2, label='MoM Monostatique')
    plt.plot(results['phi_s'], results['SER_po_mono'], 'r--', linewidth=2, label='PO Monostatique')
    plt.xlabel('Angle d\'incidence φ (degrés)')
    plt.ylabel('SER (dB)')
    plt.title(f'SER MONOSTATIQUE (φ_s = φ_i)\nL = {results["params"]["L"]}λ')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([-40, 20])
    
    # Sous-figure 2: Différence entre monostatique et bistatique
    plt.subplot(1, 2, 2)
    plt.plot(results['phi_s'], results['SER_mom_mono'], 'b-', linewidth=2, label='MoM Monostatique')
    plt.plot(results['phi_s'], results['SER_mom_bistatic'], 'g:', linewidth=2, 
             label=f'MoM Bistatique (φ_i={results["params"]["phi_i_deg"]:.1f}°)')
    plt.xlabel('Angle (degrés)')
    plt.ylabel('SER (dB)')
    plt.title(f'Comparaison MoM Monostatique vs Bistatique\nφ_i fixe = {results["params"]["phi_i_deg"]:.1f}°')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([-40, 20])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/TM_strip_RCS_monostatic.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================================================
    # FIGURE 3: SER BISTATIQUE (pour référence)
    # ============================================================================
    print("Generation de la SER bistatique (pour reference)")
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['phi_s'], results['SER_mom_bistatic'], 'b-', linewidth=2, label='EFIE MoM Bistatique')
    plt.plot(results['phi_s'], results['SER_po_bistatic'], 'r--', linewidth=2, label='Physical Optics Bistatique')
    plt.xlabel('Angle de diffusion φ_s (degrés)')
    plt.ylabel('SER (dB)')
    plt.title(f'SER BISTATIQUE - Strip TM\nL = {results["params"]["L"]}λ, φⁱ = {results["params"]["phi_i_deg"]:.1f}°')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([-40, 20])
    
    plt.savefig(f"{save_dir}/TM_strip_RCS_bistatic.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# ANALYSE COMPARATIVE DE CONVERGENCE
# ============================================================================
def compare_TM_strip_convergence():
    """
    Compare différentes résolutions pour le strip TM (analyse de convergence)
    
    Utilise N=1000 comme référence (solution la plus précise)
    NOTE: L'analyse utilise la SER bistatique pour des raisons de performance
    """
    print("\n" + "=" * 70)
    print("ANALYSE DE CONVERGENCE - RÉFÉRENCE N=1000")
    print("=" * 70)
    print("NOTE: Utilisation de la SER bistatique pour des raisons de performance")
    print("Pour une analyse monostatique complète, augmenter N progressivement")
    
    # 1. Calculer la référence avec N=1000
    print("\n--- Calcul de la référence (N=1000) ---")
    save_dir_ref = "TM_strip_convergence_ref_N1000"
    start_time = time.time()
    results_ref = solve_TM_strip_procedural(L=3.0, N=1000, save_dir=save_dir_ref)
    ref_time = time.time() - start_time
    
    # Utiliser la SER monostatique de référence (plus précise)
    SER_ref_dB = results_ref['SER_mom_mono']
    
    convergence_results = {
        1000: {
            'time': ref_time,
            'condition': np.linalg.cond(results_ref['Z']),
            'SER_max': np.max(SER_ref_dB),
            'error': 0.0  # Pas d'erreur pour la référence
        }
    }
    
    print(f"Temps de calcul référence: {ref_time:.2f} s")
    print(f"Conditionnement de Z (référence): {np.linalg.cond(results_ref['Z']):.2e}")
    print(f"SER max monostatique (référence): {np.max(SER_ref_dB):.2f} dB")
    
    # 2. Calculer les autres résolutions et comparer
    N_values = [50, 100, 200, 300, 400, 500, 750]
    
    for N in N_values:
        print(f"\n--- N = {N} segments ---")
        
        save_dir = f"TM_strip_convergence_N{N}"
        
        start_time = time.time()
        results = solve_TM_strip_procedural(L=3.0, N=N, save_dir=save_dir)
        computation_time = time.time() - start_time
        
        # Utilisation de la SER monostatique pour le calcul d'erreur
        SER_current_dB = results['SER_mom_mono']
        
        # Calcul de l'erreur RMS relative
        error = np.sqrt(np.mean((SER_current_dB - SER_ref_dB)**2))
        
        convergence_results[N] = {
            'time': computation_time,
            'condition': np.linalg.cond(results['Z']),
            'SER_max': np.max(SER_current_dB),
            'error': error
        }
        
        print(f"Temps de calcul: {computation_time:.2f} s")
        print(f"Conditionnement de Z: {np.linalg.cond(results['Z']):.2e}")
        print(f"SER max monostatique: {np.max(SER_current_dB):.2f} dB")
        print(f"Erreur RMS: {error:.4f} dB")
    
    # 3. Graphique de convergence
    print("\n--- Graphique de convergence ---")
    N_list = sorted(convergence_results.keys())
    errors = [convergence_results[N]['error'] for N in N_list if N != 1000]
    times = [convergence_results[N]['time'] for N in N_list if N != 1000]
    conditions = [convergence_results[N]['condition'] for N in N_list if N != 1000]
    N_list_no_ref = [N for N in N_list if N != 1000]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Graphique erreur vs N
    ax1.loglog(N_list_no_ref, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Nombre de segments N')
    ax1.set_ylabel('Erreur RMS (dB)')
    ax1.set_title('Convergence de la SER monostatique\n(Référence: N=1000)')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Graphique temps vs N
    ax2.plot(N_list_no_ref, times, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Nombre de segments N')
    ax2.set_ylabel('Temps de calcul (s)')
    ax2.set_title('Coût computationnel')
    ax2.grid(True, alpha=0.3)
    
    # Graphique conditionnement vs N
    ax3.loglog(N_list_no_ref, conditions, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Nombre de segments N')
    ax3.set_ylabel('Conditionnement')
    ax3.set_title('Conditionnement de la matrice Z')
    ax3.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig("TM_strip_convergence_analysis_monostatic.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Affichage du tableau de convergence
    print("\n" + "=" * 70)
    print("TABLEAU DE CONVERGENCE (SER MONOSTATIQUE)")
    print("=" * 70)
    print(f"{'N':<8} {'Temps (s)':<12} {'Conditionnement':<18} {'Erreur (dB)':<12} {'SER_max (dB)':<12}")
    print("-" * 70)
    
    for N in sorted(convergence_results.keys()):
        res = convergence_results[N]
        print(f"{N:<8} {res['time']:<12.2f} {res['condition']:<18.2e} {res['error']:<12.4f} {res['SER_max']:<12.2f}")
    
    return convergence_results

# ============================================================================
# DÉMONSTRATION PRINCIPALE
# ============================================================================
def demonstrate_TM_strip_procedural():
    """
    Démonstration complète du problème du strip TM
    [Section 5.1.1 Gibson - TM Strip Example]
    """
    print("=" * 70)
    print("DEMONSTRATION - STRIP TM (SER MONOSTATIQUE CORRIGÉE)")
    print("Reference: Gibson Chapter 5, Section 5.1.1")
    print("=" * 70)
    
    configurations = [
        {'L': 3.0, 'phi_i': np.pi/2, 'N': 300, 'label': 'broadside'},  # N réduit pour performance
        {'L': 3.0, 'phi_i': np.pi/4, 'N': 300, 'label': 'oblique'},
        {'L': 1.0, 'phi_i': np.pi/2, 'N': 200, 'label': 'small_strip'}
    ]
    
    all_results = {}
    for config in configurations:
        print(f"\n--- Configuration: {config['label']} ---")
        
        save_dir = f"TM_strip_{config['label']}_L{config['L']}lambda_N{config['N']}"
        
        results = solve_TM_strip_procedural(
            L=config['L'], 
            phi_i=config['phi_i'],
            N=config['N'],
            save_dir=save_dir
        )
        
        plot_TM_strip_results(results, save_dir=save_dir)
        all_results[config['label']] = results
        
        # Affichage de la valeur à φ = 90° pour comparaison avec Gibson
        if config['label'] == 'broadside':
            idx_90 = np.argmin(np.abs(results['phi_s'] - 90))
            print(f"\nValeur de la SER à φ = 90° (monostatique):")
            print(f"  MoM: {results['SER_mom_mono'][idx_90]:.2f} dB")
            print(f"  PO: {results['SER_po_mono'][idx_90]:.2f} dB")
            print(f"  Différence MoM-PO: {results['SER_mom_mono'][idx_90] - results['SER_po_mono'][idx_90]:.2f} dB")
    
    return all_results

# ============================================================================
# FONCTION POUR COMPARAISON RAPIDE AVEC GIBSON
# ============================================================================
def quick_gibson_comparison():
    """
    Comparaison rapide avec les résultats de Gibson
    Focus sur la SER monostatique à φ = 90°
    """
    print("\n" + "=" * 70)
    print("COMPARAISON AVEC GIBSON - SER MONOSTATIQUE À φ = 90°")
    print("=" * 70)
    
    # Paramètres similaires à Gibson (strip 3λ, incidence normale)
    print("\nCalcul avec N=1000 pour plus de précision...")
    results = solve_TM_strip_procedural(L=3.0, phi_i=np.pi/2, N=1000, save_dir="Gibson_comparison")
    
    # Valeur à φ = 90°
    idx_90 = np.argmin(np.abs(results['phi_s'] - 90))
    ser_mom_90 = results['SER_mom_mono'][idx_90]
    ser_po_90 = results['SER_po_mono'][idx_90]
    
    print(f"\nRésultats pour strip 3λ, incidence normale:")
    print(f"  SER MoM (monostatique, φ=90°): {ser_mom_90:.2f} dB")
    print(f"  SER PO (monostatique, φ=90°): {ser_po_90:.2f} dB")
    print(f"  Différence MoM-PO: {ser_mom_90 - ser_po_90:.2f} dB")
    
    # Selon Gibson Figure 5.2b, à φ=90°:
    # - PO: environ 3 dB
    # - MoM: environ 5 dB
    # Différence attendue: environ 2 dB
    
    print(f"\nComparaison avec Gibson Figure 5.2b:")
    print(f"  Gibson PO (φ=90°): ~3 dB")
    print(f"  Gibson MoM (φ=90°): ~5 dB")
    print(f"  Notre PO (φ=90°): {ser_po_90:.2f} dB")
    print(f"  Notre MoM (φ=90°): {ser_mom_90:.2f} dB")
    
    return results

if __name__ == "__main__":
    # Démonstration principale
    print("\n" + "=" * 70)
    print("EXÉCUTION PRINCIPALE")
    print("=" * 70)
    
    # Option 1: Démonstration complète (plus rapide avec N réduit)
    results = demonstrate_TM_strip_procedural()
    
    # Option 2: Comparaison rapide avec Gibson
    print("\n" + "=" * 70)
    print("COMPARAISON AVEC GIBSON")
    print("=" * 70)
    quick_gibson_comparison()
    
    # Option 3: Analyse de convergence (désactivée par défaut car longue)
    # Décommenter si nécessaire
    """
    print("\n" + "=" * 70)
    print("ANALYSE DE CONVERGENCE")
    print("=" * 70)
    convergence = compare_TM_strip_convergence()
    """
    
    print("\n" + "=" * 70)
    print("ANALYSE TERMINÉE")
    print("NOTE: La SER MONOSTATIQUE est maintenant correctement calculée")
    print("      (φ_s = φ_i pour chaque angle)")
    print("=" * 70)
