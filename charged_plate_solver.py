import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import os
import pickle

"""
================================================================================
PROBLÈME DE LA PLAQUE CHARGÉE - Référence: Gibson - Chapter 3, Section 3.1.2 (Charged Plate)
================================================================================

Ce code résout l'équation intégrale de la plaque chargée en utilisant la Méthode des Moments.
La plaque carrée de côté L est maintenue à un potentiel de 1V. L'objectif est de trouver
la distribution de charge surfacique σ(x,y) qui produit ce potentiel.

ÉQUATION INTÉGRALE [Éq. 3.16 Gibson]:
φ(𝐫) = ∬_S [σ(𝐫') / (4πε₀|𝐫-𝐫'|)] dS' = 1V

DISCRÉTISATION [Section 3.1.2 Gibson]:
- La plaque est divisée en N×N patches carrés
- On utilise des fonctions base "pulse" (charge constante par patch) [Section 3.3.1]
- On applique la méthode de point matching aux centres des patches [Section 3.2.1]

FORME MATRICIELLE [Section 3.2 Gibson]:
[Z][a] = [b]

où:
- Zₘₙ = ∬_patchₙ [1/(4πε₀|𝐫ₘ-𝐫'|)] dS'  (potentiel au patch m dû à une charge unité sur le patch n)
- bₘ = 1V  (potentiel imposé)
- aₙ = σₙ  (densité de charge sur le patch n)

CALCUL DES ÉLÉMENTS DE MATRICE [Section 3.1.2.1 Gibson]:
- Termes self (m=n): Zₘₘ = (2a/πε₀) * log(1 + √2)  (intégrale analytique sur un carré) [Éq. 3.22]
- Termes near (m≠n): Zₘₙ ≈ A/(4πε₀|𝐫ₘ-𝐫ₙ|)  (approximation centroïdale) [Éq. 3.23]

OPÉRATEUR SOLUTION [Section 3.2 Gibson]:
La solution est obtenue en résolvant le système linéaire : a = Z⁻¹ * b
Ceci donne la distribution de charge qui produit le potentiel de 1V sur toute la plaque.
"""

def create_plate_grid(L, N):
    """
    Crée la grille de discrétisation de la plaque [Section 3.1.2 Gibson]
    
    La plaque carrée de côté L est divisée en N×N patches carrés identiques.
    Retourne les coordonnées des centres de chaque patch.
    """
    patch_size = L / N
    x = np.linspace(patch_size/2, L - patch_size/2, N)
    y = np.linspace(patch_size/2, L - patch_size/2, N)
    X, Y = np.meshgrid(x, y)
    return X.flatten(), Y.flatten(), patch_size

def compute_plate_matrix_element(m, n, x_centers, y_centers, patch_size, patch_area):
    """
    Calcule l'élément de matrice Z_mn [Section 3.1.2.1 Gibson]
    
    Cette fonction implémente les équations (3.22) et (3.23) du livre Gibson
    pour le calcul des éléments de la matrice d'impédance.
    """
    epsilon_0 = 8.854187817e-12
    
    if m == n:
        # ÉQUATION (3.22) Gibson : Terme self - intégrale analytique sur un patch carré
        # ∫∫ [1/|𝐫-𝐫'|] dS' = (2a/π) * log(1 + √2) pour un carré de côté 2a
        a = patch_size / 2  # demi-côté du patch
        return (2 * a / (np.pi * epsilon_0)) * np.log(1 + np.sqrt(2))
    else:
        # ÉQUATION (3.23) Gibson : Termes near - approximation centroïdale
        # ∫∫ [1/|𝐫-𝐫'|] dS' ≈ A / |𝐫_m - 𝐫_n|
        x_m, y_m = x_centers[m], y_centers[m]
        x_n, y_n = x_centers[n], y_centers[n]
        distance = np.sqrt((x_m - x_n)**2 + (y_m - y_n)**2)
        return patch_area / (4 * np.pi * epsilon_0 * distance)

def build_plate_matrix(L, N):
    """
    Construit la matrice d'impédance Z de la plaque [Section 3.1.2 Gibson]
    
    Cette fonction assemble la matrice complète du système linéaire
    en calculant toutes les interactions entre patches.
    """
    # Création de la grille de discrétisation
    x_centers, y_centers, patch_size = create_plate_grid(L, N)
    patch_area = patch_size ** 2
    total_patches = N * N
    
    # Initialisation de la matrice Z
    Z = np.zeros((total_patches, total_patches))
    
    print(f"Construction de la matrice {total_patches}x{total_patches} [Section 3.1.2]...")
    
    # Calcul de tous les éléments de la matrice
    for m in range(total_patches):
        if m % 50 == 0:  # Affichage de progression
            print(f"  Progression: {m}/{total_patches}")
        for n in range(total_patches):
            Z[m, n] = compute_plate_matrix_element(m, n, x_centers, y_centers, patch_size, patch_area)
    
    return Z, x_centers, y_centers, patch_size, patch_area

def solve_charged_plate_procedural(L=1.0, N=30, save_dir=None):
    """
    Résout le problème de la plaque chargée - Approche procédurale
    [Section 3.1.2 Gibson]
    
    Cette fonction implémente la méthode des moments complète pour la plaque chargée :
    1. Discrétisation de la plaque en patches
    2. Construction de la matrice d'impédance Z
    3. Construction du vecteur d'excitation b
    4. Résolution du système linéaire
    
    Parameters:
    L : float - Côté de la plaque carrée (m)
    N : int - Nombre de patches par côté (total patches = N×N) - Augmenté pour meilleure résolution
    save_dir : str - Dossier de sauvegarde (auto-généré si None)
    """
    # Création du dossier de sauvegarde cohérent
    if save_dir is None:
        save_dir = f"plate_{N}_patches"
    
    print(f"Resolution de la plaque chargee [Section 3.1.2 Gibson]")
    print(f"Parametres: L={L}m, N={N} patches par cote (total {N*N} patches)")
    print(f"Sauvegarde dans: {save_dir}")
    
    # ============================================================================
    # ÉTAPE 1: CONSTRUCTION DE LA MATRICE D'IMPÉDANCE Z
    # ============================================================================
    print("1. Construction de la matrice Z [Section 3.1.2]...")
    Z, x_centers, y_centers, patch_size, patch_area = build_plate_matrix(L, N)
    
    # ============================================================================
    # ÉTAPE 2: CONSTRUCTION DU VECTEUR D'EXCITATION b
    # ============================================================================
    print("2. Construction du vecteur d'excitation b [Eq. 3.31]...")
    b = np.ones(Z.shape[0])  # Potentiel de 1V partout sur la plaque
    
    # ============================================================================
    # ÉTAPE 3: RÉSOLUTION DU SYSTÈME LINÉAIRE
    # ============================================================================
    print("3. Resolution du systeme lineaire [Section 3.4]...")
    charge_coeffs = solve(Z, b)
    
    # ============================================================================
    # ÉTAPE 4: CALCUL DES MÉTRIQUES DE PERFORMANCE
    # ============================================================================
    print("4. Calcul des metriques...")
    cond_Z = np.linalg.cond(Z)
    charge_grid = charge_coeffs.reshape((N, N))
    total_charge = np.sum(charge_grid) * patch_area
    
    print(f"   Conditionnement de Z: {cond_Z:.2e}")
    print(f"   Charge totale estimee: {total_charge:.6e} C")
    
    # Structure des résultats
    results = {
        'charge_coeffs': charge_coeffs,  # Solution : densités de charge
        'charge_grid': charge_grid,      # Solution en grille 2D
        'Z': Z,                          # Matrice d'impédance
        'b': b,                          # Vecteur d'excitation  
        'x_centers': x_centers,          # Coordonnées x des centres
        'y_centers': y_centers,          # Coordonnées y des centres
        'patch_size': patch_size,        # Taille d'un patch
        'patch_area': patch_area,        # Surface d'un patch
        'params': {
            'L': L, 'N': N, 
            'total_patches': N * N,
            'condition_number': cond_Z,
            'total_charge': total_charge,
            'save_dir': save_dir
        }
    }
    
    print("✓ Resolution terminee avec succes")
    return results

def plot_plate_results(results, save_dir=None):
    """
    Trace les résultats pour la plaque chargée [Section 3.1.2.2 Gibson]
    
    Génère les visualisations principales :
    - Distribution de charge 2D [Figure 3.6 Gibson]
    - Distribution le long de la diagonale [Figure 3.7 Gibson]
    """
    if save_dir is None:
        save_dir = results['params']['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ============================================================================
    # FIGURE 1: DISTRIBUTION 2D [FIGURE 3.6 GIBSON]
    # ============================================================================
    print("Generation de la distribution 2D [Figure 3.6]...")
    plt.figure(figsize=(10, 8))
    
    # Création de la heatmap
    im = plt.imshow(results['charge_grid'], 
                   extent=[0, results['params']['L'], 0, results['params']['L']], 
                   cmap='hot', origin='lower', aspect='equal')
    
    plt.colorbar(im, label='Densite de charge (C/m²)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Distribution de charge sur la plaque\nL={results["params"]["L"]}m, N={results["params"]["N"]}×{results["params"]["N"]}')
    
    plt.savefig(f"{save_dir}/plate_charge_2d.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================================================
    # FIGURE 2: DISTRIBUTION LE LONG DE LA DIAGONALE [FIGURE 3.7 GIBSON]
    # ============================================================================
    print("Generation de la distribution diagonale [Figure 3.7]...")
    diagonal_charge = np.array([results['charge_grid'][i, i] for i in range(results['params']['N'])])
    positions = np.linspace(0, results['params']['L'] * np.sqrt(2), results['params']['N'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(positions, diagonal_charge, 'r-', linewidth=2)
    plt.xlabel('Position le long de la diagonale (m)')
    plt.ylabel('Densite de charge (C/m²)')
    plt.title('Distribution de charge le long de la diagonale')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{save_dir}/plate_diagonal.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_plate_results(results, save_dir=None):
    """
    Sauvegarde tous les résultats de la plaque chargée
    [Section 3.1.2.2 Gibson - Solution Analysis]
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
    np.save(f"{save_dir}/charge_grid.npy", results['charge_grid'])
    np.save(f"{save_dir}/x_centers.npy", results['x_centers'])
    np.save(f"{save_dir}/y_centers.npy", results['y_centers'])
    
    # ============================================================================
    # SAUVEGARDE DES PARAMÈTRES
    # ============================================================================
    with open(f"{save_dir}/params.pkl", 'wb') as f:
        pickle.dump(results['params'], f)
    
    # ============================================================================
    # RAPPORT DÉTAILLÉ [LIEN AVEC LE CHAPITRE 3 GIBSON]
    # ============================================================================
    with open(f"{save_dir}/plate_analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT D'ANALYSE - PLAQUE CHARGE ===\n\n")
        f.write("REFERENCE: Gibson - Chapter 3, Section 3.1.2 (Charged Plate)\n\n")
        
        f.write("PARAMETRES DU PROBLEME:\n")
        f.write(f"  Cote de la plaque (L): {results['params']['L']} m\n")
        f.write(f"  Nombre de patches par cote (N): {results['params']['N']}\n")
        f.write(f"  Total patches: {results['params']['total_patches']}\n")
        f.write(f"  Taille d'un patch: {results['patch_size']} m\n")
        f.write(f"  Surface d'un patch: {results['patch_area']} m²\n\n")
        
        f.write("INFORMATIONS SUR LA MATRICE Z [Section 3.1.2.1]:\n")
        f.write(f"  Dimensions: {results['Z'].shape}\n")
        f.write(f"  Conditionnement [Eq. 3.60]: {results['params']['condition_number']:.2e}\n")
        f.write(f"  Trace: {np.trace(results['Z']):.6e}\n\n")
        
        f.write("INFORMATIONS SUR LA SOLUTION [Section 3.1.2.2]:\n")
        f.write(f"  Charge totale: {results['params']['total_charge']:.6e} C\n")
        f.write(f"  Charge moyenne: {np.mean(results['charge_grid']):.6e} C/m²\n")
        f.write(f"  Charge maximale: {np.max(results['charge_grid']):.6e} C/m²\n")
        f.write(f"  Charge minimale: {np.min(results['charge_grid']):.6e} C/m²\n")
        f.write(f"  Rapport max/min: {np.max(results['charge_grid'])/np.min(results['charge_grid']):.2f}\n\n")
        
        f.write("EQUATIONS IMPLEMENTEES:\n")
        f.write("  Eq. 3.16: phi(r) = double integral [sigma(r')/(4pi epsilon_0 |r-r'|)] dS' = 1V\n")
        f.write("  Eq. 3.22: Z_mm = (2a/pi epsilon_0) * log(1 + sqrt(2))  (terme self)\n")
        f.write("  Eq. 3.23: Z_mn ≈ A/(4pi epsilon_0 |r_m - r_n|)  (termes near)\n")
    
    print("✓ Sauvegarde terminee")

# ============================================================================
# DÉMONSTRATION PRINCIPALE - VERSION AMÉLIORÉE
# ============================================================================
def demonstrate_plate_procedural():
    """
    Démonstration complète du problème de la plaque chargée
    [Section 3.1.2 Gibson - Charged Plate Example]
    Version améliorée avec résolution augmentée et dossiers cohérents
    """
    print("=" * 70)
    print("DEMONSTRATION - PLAQUE CHARGE (Résolution Augmentée)")
    print("Reference: Gibson Chapter 3, Section 3.1.2")
    print("=" * 70)
    
    # Résolutions avec différentes résolutions augmentées
    resolutions = [20, 30, 40]  # Résolutions augmentées pour meilleure précision
    
    all_results = {}
    for N in resolutions:
        print(f"\n--- Résolution avec {N}×{N} patches (total {N*N}) ---")
        
        save_dir = f"plate_{N}_patches"
        
        # Résolution du problème
        results = solve_charged_plate_procedural(L=1.0, N=N, save_dir=save_dir)
        
        # Visualisation des résultats
        plot_plate_results(results, save_dir=save_dir)
        
        # Sauvegarde des résultats
        save_plate_results(results, save_dir=save_dir)
        
        all_results[N] = results
    
    return all_results

# ============================================================================
# FONCTION DE DÉMONSTRATION RAPIDE
# ============================================================================
def quick_plate_demo():
    """
    Démonstration rapide en 3 lignes pour tester la plaque chargée
    """
    print("Demonstration rapide de la plaque chargee...")
    results = solve_charged_plate_procedural(L=1.0, N=30, save_dir="plate_30_patches")
    plot_plate_results(results)
    save_plate_results(results)
    return results

if __name__ == "__main__":
    # Démonstration complète [Chapter 3, Section 3.1.2]
    results = demonstrate_plate_procedural()
    
    print("\n" + "=" * 70)
    print("ANALYSE TERMINEE - TOUS LES RESULTATS SAUVEGARDES")
    print("Structure des dossiers créés:")
    print("  plate_20_patches/")
    print("  plate_30_patches/")
    print("  plate_40_patches/")
    print("=" * 70)