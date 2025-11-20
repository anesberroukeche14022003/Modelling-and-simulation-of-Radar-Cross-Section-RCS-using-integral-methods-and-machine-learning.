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

def solve_TM_strip_procedural(L=3.0, freq=1e9, phi_i=np.pi/2, N=300, save_dir=None):
    """
    Résout le problème du strip TM - Approche procédurale [Section 5.1.1 Gibson]
    
    Parameters:
    L: float - Longueur du strip (en longueurs d'onde)
    freq: float - Fréquence (Hz)
    phi_i: float - Angle d'incidence (radians)
    N: int - Nombre de segments
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
    # ÉTAPE 5: CALCUL DE LA SER [Éq. 5.7]
    # ============================================================================
    print("4. Calcul de la SER...")
    phi_s = np.linspace(0, np.pi, 181)
    SER = np.zeros_like(phi_s, dtype=complex)
    
    for i, phi in enumerate(phi_s):
        integrand = J_z * np.exp(1j * k * x_centers * np.cos(phi))
        integral = np.sum(integrand) * delta_x
        SER[i] = -omega * mu_0 * np.sqrt(1j/(8*np.pi*k)) * integral
    
    SER_dB = 10 * np.log10(4 * np.pi * np.abs(SER)**2)
    
    # Structure des résultats
    results = {
        'J_z': J_z,                    # Courant MoM
        'J_po': J_po,                  # Courant Physical Optics
        'Z': Z,                        # Matrice d'impédance
        'b': b,                        # Vecteur d'excitation
        'x_centers': x_centers,        # Positions (m)
        'x_lambda': x_centers/lambda_, # Positions (λ)
        'SER': SER_dB,                 # SER en dB
        'phi_s': np.degrees(phi_s),    # Angles de diffusion (degrés)
        'params': {
            'L': L, 'L_meters': L_meters, 'freq': freq, 'lambda': lambda_,
            'phi_i': phi_i, 'phi_i_deg': np.degrees(phi_i), 'N': N,
            'k': k, 'omega': omega, 'eta': eta,
            'save_dir': save_dir
        }
    }
    
    print("✓ Resolution terminee avec succes")
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
    
    # Calcul de la SER PO pour comparaison
    phi_i = results['params']['phi_i']
    J_po_constant = (2 / results['params']['eta']) * np.sin(phi_i)
    SER_po = np.full_like(results['phi_s'], 
                        10*np.log10(np.abs(J_po_constant * results['params']['L_meters'])**2))
    
    plt.plot(results['phi_s'], results['SER'], 'b-', linewidth=2, label='EFIE MoM')
    plt.plot(results['phi_s'], SER_po, 'r--', linewidth=2, label='Physical Optics')
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
    np.save(f"{save_dir}/SER.npy", results['SER'])
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
        f.write("  Eq. 5.7: E_z^s(ρ) = -ωμ√(j/(8πk))(e^{-jkρ}/√ρ)∫J_z(x')e^{jkx'cos(φˢ)}dx'\n")
    
    print("✓ Sauvegarde terminee")

# ============================================================================
# DÉMONSTRATION PRINCIPALE
# ============================================================================
def demonstrate_TM_strip_procedural():
    """
    Démonstration complète du problème du strip TM
    [Section 5.1.1 Gibson - TM Strip Example]
    """
    print("=" * 70)
    print("DEMONSTRATION - STRIP TM")
    print("Reference: Gibson Chapter 5, Section 5.1.1")
    print("=" * 70)
    
    # Paramètres de simulation
    configurations = [
        {'L': 3.0, 'phi_i': np.pi/2, 'N': 100, 'label': 'broadside'},
        {'L': 3.0, 'phi_i': np.pi/4, 'N': 100, 'label': 'oblique'},
        {'L': 1.0, 'phi_i': np.pi/2, 'N': 50, 'label': 'small_strip'}
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
# ANALYSE COMPARATIVE
# ============================================================================
def compare_TM_strip_methods():
    """
    Compare différentes résolutions pour le strip TM
    """
    print("\n" + "=" * 70)
    print("COMPARAISON DES RESOLUTIONS")
    print("=" * 70)
    
    N_values = [50, 100, 200]
    
    for N in N_values:
        print(f"\n--- N = {N} segments ---")
        
        save_dir = f"TM_strip_comparison_N{N}"
        
        start_time = time.time()
        results = solve_TM_strip_procedural(L=3.0, N=N, save_dir=save_dir)
        computation_time = time.time() - start_time
        
        print(f"Temps de calcul: {computation_time:.2f} s")
        print(f"Conditionnement de Z: {np.linalg.cond(results['Z']):.2e}")

if __name__ == "__main__":
    # Démonstration principale [Chapter 5, Section 5.1.1]
    results = demonstrate_TM_strip_procedural()
    
    print("\n" + "=" * 70)
    print("ANALYSE TERMINEE - TOUS LES RESULTATS SAUVEGARDES")
    print("Structure des dossiers créés:")
    print("  TM_strip_broadside_L3.0lambda_N100/")
    print("  TM_strip_oblique_L3.0lambda_N100/")
    print("  TM_strip_small_strip_L1.0lambda_N50/")
    print("=" * 70)