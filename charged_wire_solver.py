import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, inv, toeplitz
import time
import os
import pickle

"""
================================================================================
PROBLÈME DU FIL CHARGÉ - Référence: Gibson - Chapter 3, Section 3.1.1 (Charged Wire)
================================================================================

ÉQUATION INTÉGRALE À RÉSOUDRE [Éq. 3.2 Gibson]:
φ(x) = ∫₀ᴸ [λ(x') / (4πε₀√((x-x')² + a²))] dx' = 1V

où:
- φ(x) = 1V est le potentiel imposé sur le fil
- λ(x') est la densité linéique de charge inconnue
- a est le rayon du fil
- L est la longueur du fil

DISCRÉTISATION AVEC MÉTHODE DES MOMENTS [Section 3.1.1 Gibson]:
On divise le fil en N segments et on utilise des fonctions base "pulse" [Éq. 3.4-3.5 Gibson].
Le système devient [Éq. 3.6-3.7 Gibson]:

∑ₙ aₙ ∫ₓₙ⁽ⁿ⁺¹⁾ [1 / (4πε₀√((xₘ-x')² + a²))] dx' = 1V  pour m=1..N

FORME MATRICIELLE [Section 3.2 Gibson]:
[Z][a] = [b]

où:
- Zₘₙ = ∫ₓₙ⁽ⁿ⁺¹⁾ [1 / (4πε₀√((xₘ-x')² + a²))] dx'  [Éq. 3.12 Gibson]
- bₘ = 1V  [Éq. 3.13 Gibson]
- aₙ = λₙ (densité de charge sur le segment n)

CALCUL ANALYTIQUE DES ÉLÉMENTS [Éq. 3.14 Gibson]:
Zₘₙ = log( [(x_b - xₘ) + √((x_b - xₘ)² + a²)] / [(x_a - xₘ) + √((x_a - xₘ)² + a²)] )

OPÉRATEUR SOLUTION [Section 3.2 Gibson]:
a = Z⁻¹ * b
"""

def compute_wire_matrix_element(x_m, x_a, x_b, a):
    """
    Calcule Zₘₙ selon l'équation (3.14) du livre Gibson
    
    Cette fonction implémente l'intégrale analytique pour l'élément de matrice
    entre le point d'observation x_m et le segment source [x_a, x_b]
    """
    term1 = (x_b - x_m) + np.sqrt((x_b - x_m)**2 + a**2)
    term2 = (x_a - x_m) + np.sqrt((x_a - x_m)**2 + a**2)
    
    if term2 <= 0:
        return 0.0
    
    return np.log(term1 / term2)

def build_wire_toeplitz_matrix(L, a, N):
    """
    Construit la matrice Toeplitz symétrique [Section 3.1.1.1 - Éq. 3.15 Gibson]
    
    La matrice Toeplitz permet de réduire les calculs en exploitant la symétrie
    du problème : Zₘₙ ne dépend que de |m-n|
    """
    # Discrétisation du domaine [Section 3.1.1 Gibson - Figure 3.2]
    delta_x = L / N
    x_centers = np.linspace(delta_x/2, L - delta_x/2, N)
    
    # Calcul de la première ligne seulement (symétrie Toeplitz)
    first_row = np.zeros(N)
    for n in range(N):
        x_a = n * delta_x      # Début du segment source
        x_b = (n + 1) * delta_x  # Fin du segment source  
        first_row[n] = compute_wire_matrix_element(x_centers[0], x_a, x_b, a)
    
    # Construction de la matrice Toeplitz complète
    Z = toeplitz(first_row)
    return Z, x_centers, delta_x

def solve_charged_wire_procedural(L=1.0, a=1e-3, N=200, method='implicit', save_dir=None):
    """
    Résout le problème du fil chargé - Approche procédurale
    [Section 3.1.1 Gibson]
    
    Cette fonction encapsule toute la méthode des moments pour le fil chargé :
    1. Construction de la matrice d'impédance Z
    2. Construction du vecteur d'excitation b  
    3. Résolution du système linéaire
    4. Retour des résultats complets
    
    Parameters:
    L : float - Longueur du fil (m)
    a : float - Rayon du fil (m) 
    N : int - Nombre de segments de discrétisation (augmenté pour meilleure résolution)
    method : str - 'implicit' (solve direct) ou 'explicit' (inversion de matrice)
    save_dir : str - Dossier de sauvegarde (auto-généré si None)
    """
    # Création du dossier de sauvegarde cohérent
    if save_dir is None:
        save_dir = f"wire_{N}_segments"
    
    epsilon_0 = 8.854187817e-12
    
    print(f"Resolution du fil charge [Section 3.1.1 Gibson]")
    print(f"Parametres: L={L}m, a={a}m, N={N} segments")
    print(f"Sauvegarde dans: {save_dir}")
    
    # ============================================================================
    # ÉTAPE 1: CONSTRUCTION DE LA MATRICE D'IMPÉDANCE Z
    # ============================================================================
    print("1. Construction de la matrice Z [Eq. 3.12-3.15]...")
    Z, x_centers, delta_x = build_wire_toeplitz_matrix(L, a, N)
    
    # ============================================================================
    # ÉTAPE 2: CONSTRUCTION DU VECTEUR D'EXCITATION b  
    # ============================================================================
    print("2. Construction du vecteur d'excitation b [Eq. 3.13]...")
    b = 4 * np.pi * epsilon_0 * np.ones(N)  # Potentiel de 1V partout
    
    # ============================================================================
    # ÉTAPE 3: RÉSOLUTION DU SYSTÈME LINÉAIRE
    # ============================================================================
    print(f"3. Resolution du systeme lineaire [Section 3.4]...")
    if method == 'explicit':
        # Méthode explicite : calcul de Z⁻¹ [Section 3.4.2 - LU Decomposition]
        print("   Methode explicite : calcul de Z⁻¹")
        Z_inv = inv(Z)
        charge_coeffs = Z_inv @ b
    else:
        # Méthode implicite : résolution directe [Section 3.4.1 - Gaussian Elimination]  
        print("   Methode implicite : resolution directe")
        charge_coeffs = solve(Z, b)
        Z_inv = None
    
    # ============================================================================
    # ÉTAPE 4: PRÉPARATION DES RÉSULTATS
    # ============================================================================
    print("4. Preparation des resultats...")
    
    # Calcul de métriques importantes [Section 3.4.3 - Condition Number]
    cond_Z = np.linalg.cond(Z)
    total_charge = np.sum(charge_coeffs * delta_x)
    
    print(f"   Conditionnement de Z: {cond_Z:.2e}")
    print(f"   Charge totale estimee: {total_charge:.6e} C")
    
    # Structure des résultats
    results = {
        'charge_coeffs': charge_coeffs,  # Solution a = [a₁, a₂, ..., a_N]
        'Z': Z,                          # Matrice d'impédance 
        'Z_inv': Z_inv,                  # Inverse de Z (si méthode explicite)
        'b': b,                          # Vecteur d'excitation
        'x_centers': x_centers,          # Positions des centres des segments
        'delta_x': delta_x,              # Longueur des segments
        'params': {
            'L': L, 'a': a, 'N': N, 
            'epsilon_0': epsilon_0,
            'condition_number': cond_Z,
            'total_charge': total_charge,
            'save_dir': save_dir
        }
    }
    
    print("Resolution terminee avec succes")
    return results

def plot_wire_results(results, save_dir=None):
    """
    Trace les résultats pour le fil chargé [Section 3.1.1.2 Gibson]
    
    Cette fonction génère les visualisations principales :
    - Distribution de charge le long du fil [Figure 3.3 Gibson]
    - Structure des opérateurs Z et Z⁻¹
    - Vérification de la solution
    """
    if save_dir is None:
        save_dir = results['params']['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ============================================================================
    # FIGURE 1: DISTRIBUTION DE CHARGE [FIGURE 3.3 GIBSON]
    # ============================================================================
    print("Generation de la distribution de charge [Figure 3.3]...")
    plt.figure(figsize=(10, 6))
    plt.plot(results['x_centers'], results['charge_coeffs'], 'b-', linewidth=2, label='Solution MoM')
    plt.xlabel('Position le long du fil (m)')
    plt.ylabel('Densite de charge (C/m)')
    plt.title(f'Distribution de charge sur le fil charge\nL={results["params"]["L"]}m, N={results["params"]["N"]} segments')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{save_dir}/wire_charge_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================================================
    # FIGURE 2: OPÉRATEURS Z ET Z⁻¹ [SECTION 3.1.1.1]
    # ============================================================================
    if results['Z_inv'] is not None:
        print("Generation des visualisations des operateurs...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Affichage d'une sous-matrice pour la clarté
        display_size = min(10, results['params']['N'])
        
        im1 = ax1.imshow(results['Z'][:display_size, :display_size], cmap='RdBu_r')
        ax1.set_title(f'Matrice Z (premiers {display_size}x{display_size} elements)\n[Section 3.1.1.1 Gibson]')
        ax1.set_xlabel('Indice du segment source n')
        ax1.set_ylabel('Indice du segment observation m')
        plt.colorbar(im1, ax=ax1, label='Z_mn')
        
        im2 = ax2.imshow(results['Z_inv'][:display_size, :display_size], cmap='RdBu_r')
        ax2.set_title(f'Matrice Z⁻¹ (premiers {display_size}x{display_size} elements)\n[Section 3.4.2 Gibson]')
        ax2.set_xlabel('Indice du segment source n')
        ax2.set_ylabel('Indice du segment observation m')
        plt.colorbar(im2, ax=ax2, label='(Z⁻¹)_mn')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/wire_operators.png", dpi=300, bbox_inches='tight')
        plt.show()

def verify_wire_solution(results):
    """
    Vérifie la solution en recalculant le potentiel [Section 3.1.1.2 Gibson]
    
    Cette étape cruciale valide la solution en recalculant le potentiel
    à partir de la distribution de charge obtenue et en comparant avec 
    le potentiel imposé de 1V.
    """
    save_dir = results['params']['save_dir']
    
    print("Verification de la solution [Section 3.1.1.2]...")
    
    # Recalcul du potentiel à partir de la solution [Éq. 3.2 Gibson]
    potential_verify = np.zeros(results['params']['N'])
    for m in range(results['params']['N']):
        for n in range(results['params']['N']):
            potential_verify[m] += results['charge_coeffs'][n] * results['Z'][m, n]
    
    # Normalisation [Éq. 3.2 Gibson]
    potential_verify /= (4 * np.pi * results['params']['epsilon_0'])
    
    # Calcul de l'erreur
    error_rms = np.sqrt(np.mean((potential_verify - 1.0)**2))
    max_error = np.max(np.abs(potential_verify - 1.0))
    
    print(f"   Erreur RMS: {error_rms:.6f} V")
    print(f"   Erreur maximale: {max_error:.6f} V")
    
    # Graphique de vérification
    plt.figure(figsize=(10, 6))
    plt.plot(results['x_centers'], potential_verify, 'r-', linewidth=2, 
             label='Potentiel recalculé à partir de la solution')
    plt.axhline(y=1.0, color='k', linestyle='--', linewidth=2, 
                label='Potentiel impose (1V)')
    plt.xlabel('Position le long du fil (m)')
    plt.ylabel('Potentiel electrique (V)')
    plt.title('Verification de la solution\nComparaison du potentiel recalcule vs impose')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0.8, 1.2])
    
    # Annotation des erreurs
    plt.text(0.05, 0.95, f'Erreur RMS: {error_rms:.4f} V\nErreur max: {max_error:.4f} V', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    plt.savefig(f"{save_dir}/wire_solution_verification.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return error_rms, max_error

def save_wire_results(results, save_dir=None):
    """
    Sauvegarde tous les résultats dans des fichiers pour analyse ultérieure
    [Section 3.1.1.2 Gibson - Solution Analysis]
    """
    if save_dir is None:
        save_dir = results['params']['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Sauvegarde des resultats dans {save_dir}/...")
    
    # ============================================================================
    # SAUVEGARDE DES DONNÉES NUMÉRIQUES
    # ============================================================================
    np.save(f"{save_dir}/Z_matrix.npy", results['Z'])
    np.save(f"{save_dir}/b_vector.npy", results['b'])
    np.save(f"{save_dir}/charge_coeffs.npy", results['charge_coeffs'])
    np.save(f"{save_dir}/x_centers.npy", results['x_centers'])
    
    if results['Z_inv'] is not None:
        np.save(f"{save_dir}/Z_inv.npy", results['Z_inv'])
    
    # ============================================================================
    # SAUVEGARDE DES PARAMÈTRES ET MÉTRIQUES
    # ============================================================================
    with open(f"{save_dir}/params.pkl", 'wb') as f:
        pickle.dump(results['params'], f)
    
    # ============================================================================
    # RAPPORT DÉTAILLÉ [LIEN AVEC LE CHAPITRE 3 GIBSON]
    # ============================================================================
    with open(f"{save_dir}/wire_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT D'ANALYSE - FIL CHARGE ===\n\n")
        f.write("REFERENCE: Gibson - Chapter 3, Section 3.1.1 (Charged Wire)\n\n")
        
        f.write("PARAMETRES DU PROBLEME:\n")
        f.write(f"  Longueur du fil (L): {results['params']['L']} m\n")
        f.write(f"  Rayon du fil (a): {results['params']['a']} m\n")
        f.write(f"  Nombre de segments (N): {results['params']['N']}\n")
        f.write(f"  Longueur des segments (delta_x): {results['delta_x']} m\n")
        f.write(f"  epsilon_0: {results['params']['epsilon_0']} F/m\n\n")
        
        f.write("INFORMATIONS SUR LA MATRICE Z [Section 3.1.1.1]:\n")
        f.write(f"  Dimensions: {results['Z'].shape}\n")
        f.write(f"  Conditionnement [Eq. 3.60]: {results['params']['condition_number']:.2e}\n")
        f.write(f"  Trace: {np.trace(results['Z']):.6e}\n")
        f.write(f"  Norme Frobenius: {np.linalg.norm(results['Z'], 'fro'):.6e}\n\n")
        
        f.write("INFORMATIONS SUR LA SOLUTION [Section 3.1.1.2]:\n")
        f.write(f"  Charge totale: {results['params']['total_charge']:.6e} C\n")
        f.write(f"  Charge moyenne: {np.mean(results['charge_coeffs']):.6e} C/m\n")
        f.write(f"  Charge maximale: {np.max(results['charge_coeffs']):.6e} C/m\n")
        f.write(f"  Charge minimale: {np.min(results['charge_coeffs']):.6e} C/m\n")
        f.write(f"  Rapport max/min: {np.max(results['charge_coeffs'])/np.min(results['charge_coeffs']):.2f}\n\n")
        
        f.write("EQUATIONS IMPLEMENTEES:\n")
        f.write("  Eq. 3.2: phi(x) = integral [lambda(x')/(4pi epsilon_0 |r-r'|)] dx' = 1V\n")
        f.write("  Eq. 3.14: Z_mn = log[(x_b-x_m+sqrt((x_b-x_m)^2+a^2))/(x_a-x_m+sqrt((x_a-x_m)^2+a^2))]\n")
        f.write("  Eq. 3.15: Structure Toeplitz de la matrice Z\n")
    
    print("Sauvegarde terminee")

# ============================================================================
# DÉMONSTRATION PRINCIPALE - VERSION AMÉLIORÉE
# ============================================================================
def demonstrate_wire_procedural():
    """
    Démonstration complète du problème du fil chargé
    [Section 3.1.1 Gibson - Charged Wire Example]
    Version améliorée avec résolution augmentée et dossiers cohérents
    """
    print("=" * 70)
    print("DEMONSTRATION - FIL CHARGE (Résolution Augmentée)")
    print("Reference: Gibson Chapter 3, Section 3.1.1")
    print("=" * 70)
    
    # Résolutions avec différents niveaux de discrétisation
    N_values = [100, 200, 500]  # Résolutions augmentées pour meilleure précision
    
    all_results = {}
    for N in N_values:
        print(f"\n--- Résolution avec N={N} segments ---")
        
        # ============================================================================
        # ÉTAPE 1: RÉSOLUTION DU PROBLÈME
        # ============================================================================
        save_dir = f"wire_{N}_segments"
        results = solve_charged_wire_procedural(L=1.0, a=1e-3, N=N, method='implicit', save_dir=save_dir)
        
        # ============================================================================
        # ÉTAPE 2: VISUALISATION DES RÉSULTATS  
        # ============================================================================
        print("\n2. VISUALISATION [Section 3.1.1.2]")
        plot_wire_results(results, save_dir=save_dir)
        
        # ============================================================================
        # ÉTAPE 3: VÉRIFICATION DE LA SOLUTION
        # ============================================================================
        print("\n3. VERIFICATION [Section 3.1.1.2]")
        error_rms, max_error = verify_wire_solution(results)
        
        # ============================================================================
        # ÉTAPE 4: SAUVEGARDE DES RÉSULTATS
        # ============================================================================
        print("\n4. SAUVEGARDE POUR ANALYSE")
        save_wire_results(results, save_dir=save_dir)
        
        all_results[N] = results
        print(f"Simulation N={N} terminée - Erreur RMS: {error_rms:.6f} V")
    
    return all_results

# ============================================================================
# COMPARAISON DES MÉTHODES DE RÉSOLUTION - VERSION AMÉLIORÉE
# ============================================================================
def compare_wire_methods():
    """
    Compare les méthodes de résolution pour le fil chargé
    [Section 3.4 Gibson - Solution of Matrix Equations]
    Version améliorée avec dossiers cohérents
    """
    print("\n" + "=" * 70)
    print("COMPARAISON DES METHODES DE RESOLUTION")
    print("Reference: Gibson Chapter 3, Section 3.4")
    print("=" * 70)
    
    sizes = [50, 100, 200]  # Différents niveaux de discrétisation [Section 3.3.5]
    
    for N in sizes:
        print(f"\n--- N = {N} segments [Section 3.3.5] ---")
        
        # Méthode implicite [Section 3.4.1 - Gaussian Elimination]
        save_dir_implicit = f"wire_{N}_segments_implicit"
        start_time = time.time()
        results_implicit = solve_charged_wire_procedural(L=1.0, a=1e-3, N=N, method='implicit', save_dir=save_dir_implicit)
        time_implicit = time.time() - start_time
        
        # Méthode explicite [Section 3.4.2 - LU Decomposition]
        save_dir_explicit = f"wire_{N}_segments_explicit"
        start_time = time.time()
        results_explicit = solve_charged_wire_procedural(L=1.0, a=1e-3, N=N, method='explicit', save_dir=save_dir_explicit)
        time_explicit = time.time() - start_time
        
        # Comparaison
        diff = np.max(np.abs(results_implicit['charge_coeffs'] - results_explicit['charge_coeffs']))
        
        print(f"Temps implicite [3.4.1]: {time_implicit:.4f} s")
        print(f"Temps explicite [3.4.2]: {time_explicit:.4f} s") 
        print(f"Difference solutions: {diff:.2e}")

if __name__ == "__main__":
    # Démonstration principale [Chapter 3, Section 3.1.1]
    results = demonstrate_wire_procedural()
    
    # Comparaison des méthodes [Chapter 3, Section 3.4]
    compare_wire_methods()
    
    print("\n" + "=" * 70)
    print("ANALYSE TERMINEE - TOUS LES RESULTATS SAUVEGARDES")
    print("Structure des dossiers créés:")
    print("  wire_100_segments/")
    print("  wire_200_segments/") 
    print("  wire_500_segments/")
    print("  wire_50_segments_implicit/")
    print("  wire_100_segments_explicit/")

    print("=" * 70)
