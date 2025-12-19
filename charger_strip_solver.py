import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, toeplitz
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
    
    # CORRECTION: Initialisation avec des zéros complexes
    first_row = np.zeros(N, dtype=complex)
    for n in range(N):
        first_row[n] = compute_TM_strip_matrix_element(0, n, x_centers, delta_x, k)
    
    # Construction de la matrice Toeplitz
    Z = toeplitz(first_row)
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

def solve_TM_strip_procedural(L=3.0, freq=1e9, phi_i=np.pi/2, N=500, save_dir=None):
    """
    Résout le problème du strip TM - Approche procédurale [Section 5.1.1 Gibson]
    
    Parameters:
    L: float - Longueur du strip (en longueurs d'onde)
    freq: float - Fréquence (Hz)
    phi_i: float - Angle d'incidence (radians)
    N: int - Nombre de segments (augmenté pour meilleure convergence)
    save_dir: str - Dossier de sauvegarde
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
    
    # ============================================================================
    # ÉTAPE 2: CALCUL DU CHAMP INCIDENT ET VECTEUR D'EXCITATION
    # ============================================================================
    print("2. Calcul du champ incident [Eq. 5.1]...")
    E_z_i = compute_incident_field(x_centers, k, phi_i)
    b = compute_excitation_vector(E_z_i, omega, mu_0)
    
    # ============================================================================
    # ÉTAPE 3: RÉSOLUTION DU SYSTÈME LINÉAIRE
    # ============================================================================
    print("3. Resolution du systeme lineaire [Section 3.4]...")
    J_z = solve(Z, b)
    
    # ============================================================================
    # ÉTAPE 4: CALCUL DU COURANT PHYSICAL OPTICS POUR COMPARAISON [Éq. 5.8]
    # ============================================================================
    J_po = (2 / eta) * np.sin(phi_i) * E_z_i
    
    # ============================================================================
    # ÉTAPE 5: CALCUL DE LA SER POUR MoM ET PO [CORRIGÉ SELON LES COMMENTAIRES]
    # ============================================================================
    print("4. Calcul de la SER (MoM et PO)...")
    phi_s = np.linspace(0, np.pi, 181)
    SER_mom = np.zeros_like(phi_s, dtype=complex)
    SER_po = np.zeros_like(phi_s, dtype=complex)
    
    for i, phi in enumerate(phi_s):
        # SER MoM (comme avant)
        integrand_mom = J_z * np.exp(1j * k * x_centers * np.cos(phi))
        integral_mom = np.sum(integrand_mom) * delta_x
        SER_mom[i] = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k)) * integral_mom
        
        # SER PO (MÊME TRAITEMENT QUE MoM mais avec J_po)
        integrand_po = J_po * np.exp(1j * k * x_centers * np.cos(phi))
        integral_po = np.sum(integrand_po) * delta_x
        SER_po[i] = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k)) * integral_po
    
    # CORRECTION : facteur 2*pi au lieu de 4*pi pour le 2D
    SER_mom_dB = 10 * np.log10(2 * np.pi * np.abs(SER_mom)**2)
    SER_po_dB = 10 * np.log10(2 * np.pi * np.abs(SER_po)**2)
    
    # Structure des résultats
    results = {
        'J_z': J_z,                    # Courant MoM
        'J_po': J_po,                  # Courant Physical Optics
        'Z': Z,                        # Matrice d'impédance
        'b': b,                        # Vecteur d'excitation
        'x_centers': x_centers,        # Positions (m)
        'x_lambda': x_centers/lambda_, # Positions (λ)
        'SER_mom': SER_mom_dB,         # SER MoM en dB (2*pi)
        'SER_po': SER_po_dB,           # SER PO en dB (2*pi)
        'phi_s': np.degrees(phi_s),    # Angles de diffusion (degrés)
        'params': {
            'L': L, 'L_meters': L_meters, 'freq': freq, 'lambda': lambda_,
            'phi_i': phi_i, 'phi_i_deg': np.degrees(phi_i), 'N': N,
            'k': k, 'omega': omega, 'eta': eta,
            'save_dir': save_dir
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
    print("Generation du graphique de courant [Figure 5.2a]...")
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
    # FIGURE 2: SER MONOSTATIQUE [FIGURE 5.2b GIBSON]
    # ============================================================================
    print("Generation de la SER [Figure 5.2b]...")
    plt.figure(figsize=(10, 6))
    
    # CORRECTION : Utilisation des SER calculées avec le même traitement
    plt.plot(results['phi_s'], results['SER_mom'], 'b-', linewidth=2, label='EFIE MoM')
    plt.plot(results['phi_s'], results['SER_po'], 'r--', linewidth=2, label='Physical Optics')
    plt.xlabel('Angle de diffusion φ (degrés)')
    plt.ylabel('SER (dB)')
    plt.title(f'SER Monostatique - Strip TM\nL = {results["params"]["L"]}λ, φⁱ = {results["params"]["phi_i_deg"]:.1f}°')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([-40, 20])
    
    plt.savefig(f"{save_dir}/TM_strip_RCS.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_TM_strip_results(results, save_dir=None):
    """
    Sauvegarde tous les résultats du strip TM
    """
    if save_dir is None:
        save_dir = results['params']['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Sauvegarde des resultats dans {save_dir}/...")
    
    # Sauvegarde des données numériques
    np.save(f"{save_dir}/Z_matrix.npy", results['Z'])
    np.save(f"{save_dir}/J_z.npy", results['J_z'])
    np.save(f"{save_dir}/J_po.npy", results['J_po'])
    np.save(f"{save_dir}/x_centers.npy", results['x_centers'])
    np.save(f"{save_dir}/SER_mom.npy", results['SER_mom'])
    np.save(f"{save_dir}/SER_po.npy", results['SER_po'])
    np.save(f"{save_dir}/phi_s.npy", results['phi_s'])
    
    # Sauvegarde des paramètres
    with open(f"{save_dir}/params.pkl", 'wb') as f:
        pickle.dump(results['params'], f)
    
    # Rapport détaillé
    with open(f"{save_dir}/TM_strip_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT D'ANALYSE - STRIP TM ===\n\n")
        f.write("REFERENCE: Gibson - Chapter 5, Section 5.1.1 (TM Strip)\n\n")
        
        f.write("PARAMETRES DU PROBLEME:\n")
        f.write(f"  Longueur du strip: {results['params']['L']} λ\n")
        f.write(f"  Longueur du strip: {results['params']['L_meters']:.3f} m\n")
        f.write(f"  Fréquence: {results['params']['freq']/1e9:.3f} GHz\n")
        f.write(f"  Longueur d'onde: {results['params']['lambda']:.3f} m\n")
        f.write(f"  Angle d'incidence: {results['params']['phi_i_deg']:.1f}°\n")
        f.write(f"  Nombre de segments: {results['params']['N']}\n")
        f.write(f"  Nombre d'onde k: {results['params']['k']:.3f} rad/m\n")
        f.write(f"  Impédance η: {results['params']['eta']:.1f} Ω\n\n")
        
        f.write("EQUATIONS IMPLEMENTEES:\n")
        f.write("  Eq. 5.5: ∫J_z(x')H₀⁽²⁾(k|x-x'|)dx' = (4/(ωμ))E_zⁱ(x)\n")
        f.write("  Eq. 5.15: Z_mm = Δ_x[1 - j(2/π)log(γkΔ_x/(4e))]\n")
        f.write("  Eq. 5.16: Z_mn = Δ_x[1 - j(1/π)(3log(3γkΔ_x/4)-log(γkΔ_x/4)-2)]\n")
        f.write("  Eq. 5.11: Z_mn ≈ Δ_x H₀⁽²⁾(k|x_m-x_n|) (far terms)\n")
        f.write("  Eq. 5.8: J_po = (2/η)sin(φⁱ)e^{jkx cos(φⁱ)}\n")
        f.write("  Eq. 5.7: E_z^s(ρ) = -ωμ√(j/(8πk))(e^{-jkρ}/√ρ)∫J_z(x')e^{jkx'cos(φˢ)}dx'\n\n")
        
        f.write("CORRECTIONS APPLIQUÉES:\n")
        f.write("  - SER PO calculée avec la même intégration que MoM\n")
        f.write("  - Facteur 2*π utilisé au lieu de 4*π pour le 2D\n")
        f.write("  - N augmenté à 500 segments pour meilleure convergence\n")
    
    print("Sauvegarde terminee")

# ============================================================================
# DÉMONSTRATION PRINCIPALE (N AUGMENTÉ)
# ============================================================================
def demonstrate_TM_strip_procedural():
    """
    Démonstration complète du problème du strip TM
    [Section 5.1.1 Gibson - TM Strip Example]
    """
    print("=" * 70)
    print("DEMONSTRATION - STRIP TM (Convergence améliorée)")
    print("Reference: Gibson Chapter 5, Section 5.1.1")
    print("=" * 70)
    
    # Paramètres de simulation avec N augmenté
    configurations = [
        {'L': 3.0, 'phi_i': np.pi/2, 'N': 500, 'label': 'broadside'},
        {'L': 3.0, 'phi_i': np.pi/4, 'N': 500, 'label': 'oblique'},
        {'L': 1.0, 'phi_i': np.pi/2, 'N': 300, 'label': 'small_strip'}
    ]
    
    all_results = {}
    for config in configurations:
        print(f"\n--- Configuration: {config['label']} ---")
        
        save_dir = f"TM_strip_{config['label']}_L{config['L']}lambda_N{config['N']}"
        
        # Résolution du problème
        results = solve_TM_strip_procedural(
            L=config['L'], 
            phi_i=config['phi_i'],
            N=config['N'],
            save_dir=save_dir
        )
        
        # Visualisation des résultats
        plot_TM_strip_results(results, save_dir=save_dir)
        
        # Sauvegarde des résultats
        save_TM_strip_results(results, save_dir=save_dir)
        
        all_results[config['label']] = results
    
    return all_results

# ============================================================================
# ANALYSE COMPARATIVE DE CONVERGENCE (CORRIGÉE)
# ============================================================================
def compare_TM_strip_convergence():
    """
    Compare différentes résolutions pour le strip TM (analyse de convergence)
    """
    print("\n" + "=" * 70)
    print("ANALYSE DE CONVERGENCE")
    print("=" * 70)
    
    # 1. D'abord, calculer la référence (N=500)
    print("\n--- Calcul de la référence (N=500) ---")
    save_dir_ref = "TM_strip_convergence_ref_N500"
    start_time = time.time()
    results_ref = solve_TM_strip_procedural(L=3.0, N=500, save_dir=save_dir_ref)
    ref_time = time.time() - start_time
    reference = results_ref['SER_mom']
    
    convergence_results = {
        500: {
            'time': ref_time,
            'condition': np.linalg.cond(results_ref['Z']),
            'SER_max': np.max(results_ref['SER_mom']),
            'error': 0.0  # Pas d'erreur pour la référence
        }
    }
    
    print(f"Temps de calcul référence: {ref_time:.2f} s")
    print(f"Conditionnement de Z (référence): {np.linalg.cond(results_ref['Z']):.2e}")
    
    # 2. Ensuite, calculer les autres résolutions et comparer
    N_values = [100, 200, 300, 400]
    
    for N in N_values:
        print(f"\n--- N = {N} segments ---")
        
        save_dir = f"TM_strip_convergence_N{N}"
        
        start_time = time.time()
        results = solve_TM_strip_procedural(L=3.0, N=N, save_dir=save_dir)
        computation_time = time.time() - start_time
        
        # Interpolation pour comparer aux mêmes angles
        SER_current = results['SER_mom']
        error = np.mean(np.abs(SER_current - reference))
        
        convergence_results[N] = {
            'time': computation_time,
            'condition': np.linalg.cond(results['Z']),
            'SER_max': np.max(results['SER_mom']),
            'error': error
        }
        
        print(f"Temps de calcul: {computation_time:.2f} s")
        print(f"Conditionnement de Z: {np.linalg.cond(results['Z']):.2e}")
        print(f"Erreur relative: {error:.4f} dB")
    
    # 3. Graphique de convergence
    print("\n--- Graphique de convergence ---")
    N_list = sorted(convergence_results.keys())
    errors = [convergence_results[N]['error'] for N in N_list]
    times = [convergence_results[N]['time'] for N in N_list]
    conditions = [convergence_results[N]['condition'] for N in N_list]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Graphique erreur vs N
    ax1.semilogy(N_list[:-1], errors[:-1], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Nombre de segments N')
    ax1.set_ylabel('Erreur RMS (dB) - échelle log')
    ax1.set_title('Convergence de la SER')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Graphique temps vs N
    ax2.plot(N_list, times, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Nombre de segments N')
    ax2.set_ylabel('Temps de calcul (s)')
    ax2.set_title('Coût computationnel')
    ax2.grid(True, alpha=0.3)
    
    # Graphique conditionnement vs N
    ax3.semilogy(N_list, conditions, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Nombre de segments N')
    ax3.set_ylabel('Conditionnement - échelle log')
    ax3.set_title('Conditionnement de la matrice Z')
    ax3.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig("TM_strip_convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Affichage du tableau de convergence
    print("\n" + "=" * 70)
    print("TABLEAU DE CONVERGENCE")
    print("=" * 70)
    print(f"{'N':<8} {'Temps (s)':<12} {'Conditionnement':<18} {'Erreur (dB)':<12} {'SER_max (dB)':<12}")
    print("-" * 70)
    
    for N in sorted(convergence_results.keys()):
        res = convergence_results[N]
        print(f"{N:<8} {res['time']:<12.2f} {res['condition']:<18.2e} {res['error']:<12.4f} {res['SER_max']:<12.2f}")
    
    return convergence_results

if __name__ == "__main__":
    # Démonstration principale [Chapter 5, Section 5.1.1]
    results = demonstrate_TM_strip_procedural()
    
    # Analyse de convergence
    convergence = compare_TM_strip_convergence()
    
    print("\n" + "=" * 70)
    print("ANALYSE TERMINEE - TOUS LES RESULTATS SAUVEGARDES")
    print("Structure des dossiers créés:")
    print("  TM_strip_broadside_L3.0lambda_N500/")
    print("  TM_strip_oblique_L3.0lambda_N500/")
    print("  TM_strip_small_strip_L1.0lambda_N300/")
    print("  TM_strip_convergence_ref_N500/")
    print("  TM_strip_convergence_N100/")
    print("  TM_strip_convergence_N200/")
    print("  TM_strip_convergence_N300/")
    print("  TM_strip_convergence_N400/")
    print("=" * 70)
