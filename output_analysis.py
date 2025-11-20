import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from scipy.linalg import solve

class CompletePlateAnalysis:
    """
    ANALYSE COMPLÈTE DE LA PLAQUE CHARGÉE
    Référence: Gibson - Chapter 3, Section 3.1.2
    """
    
    def __init__(self, results_dir="plate_30_patches"):
        self.results_dir = results_dir
        self.load_all_data()
        
    def load_all_data(self):
        """Charge TOUS les fichiers et explique leur contenu"""
        print("=" * 70)
        print("CHARGEMENT ET EXPLICATION DES FICHIERS")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        files_info = {
            'Z_matrix.npy': "MATRICE D'IMPÉDANCE - Interactions entre tous les patches",
            'charge_grid.npy': "SOLUTION - Densités de charge σ en grille 2D", 
            'x_centers.npy': "POSITIONS X - Coordonnées x des centres des patches",
            'y_centers.npy': "POSITIONS Y - Coordonnées y des centres des patches",
            'b_vector.npy': "VECTEUR EXCITATION - Potentiel imposé de 1V partout",
            'charge_coeffs.npy': "SOLUTION VECTORISÉE - Densités de charge en 1D",
            'params.pkl': "PARAMÈTRES - Tous les paramètres de simulation"
        }
        
        self.data = {}
        self.explanations = {}
        
        for file, explanation in files_info.items():
            try:
                filepath = f"{self.results_dir}/{file}"
                if file.endswith('.npy'):
                    self.data[file] = np.load(filepath)
                elif file.endswith('.pkl'):
                    with open(filepath, 'rb') as f:
                        self.data[file] = pickle.load(f)
                
                self.explanations[file] = explanation
                print(f"{file:20} -> {explanation}")
                
            except Exception as e:
                print(f"{file:20} -> Erreur: {e}")
        
        self.Z = self.data.get('Z_matrix.npy')
        self.charge_grid = self.data.get('charge_grid.npy') 
        self.x_centers = self.data.get('x_centers.npy')
        self.y_centers = self.data.get('y_centers.npy')
        self.b = self.data.get('b_vector.npy')
        self.charge_coeffs = self.data.get('charge_coeffs.npy')
        self.params = self.data.get('params.pkl')
        
        if self.params:
            self.N = self.params.get('N', 0)
            self.L = self.params.get('L', 1.0)
        
        print(f"Donnees chargees: {self.N}x{self.N} patches, L={self.L}m")
        print(f"Dossier d'analyse: {self.results_dir}")
    
    def show_data_summary(self):
        """Affiche un résumé complet de toutes les données"""
        print("\n" + "=" * 70)
        print("RESUME COMPLET DES DONNEES")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        for file, data in self.data.items():
            print(f"\n{file} : {self.explanations[file]}")
            
            if isinstance(data, np.ndarray):
                print(f"   Shape: {data.shape}, Type: {data.dtype}")
                print(f"   Min: {np.min(data):.2e}, Max: {np.max(data):.2e}, Mean: {np.mean(data):.2e}")
    
    def create_essential_visualizations(self):
        """Crée les visualisations essentielles demandées"""
        print("\n" + "=" * 70)
        print("GENERATION DES VISUALISATIONS ESSENTIELLES")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        # Configuration pour 2x3 subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Distribution 3D
        ax1 = fig.add_subplot(231, projection='3d')
        X, Y = np.meshgrid(np.linspace(0, self.L, self.N), np.linspace(0, self.L, self.N))
        surf = ax1.plot_surface(X, Y, self.charge_grid, cmap='hot', alpha=0.9, 
                               linewidth=0, antialiased=True)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('σ (C/m²)')
        ax1.set_title('Distribution 3D de la charge')
        fig.colorbar(surf, ax=ax1, shrink=0.6, label='Densité de charge (C/m²)')
        
        # 2. Carte de courant 2D
        ax2 = fig.add_subplot(232)
        im = ax2.imshow(self.charge_grid, extent=[0, self.L, 0, self.L], 
                       cmap='hot', origin='lower', aspect='equal')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Carte de courant 2D')
        plt.colorbar(im, ax=ax2, label='σ (C/m²)')
        
        # 3. Matrice Z (log scale)
        ax3 = fig.add_subplot(233)
        display_size = min(20, self.N)
        im_z = ax3.imshow(np.log10(np.abs(self.Z[:display_size, :display_size])), cmap='viridis')
        ax3.set_xlabel('Index patch n')
        ax3.set_ylabel('Index patch m')
        ax3.set_title(f'Matrice Z (log|Zₘₙ| - {display_size}x{display_size})')
        plt.colorbar(im_z, ax=ax3, label='log₁₀|Zₘₙ|')
        
        # 4. Distribution le long de la diagonale
        ax4 = fig.add_subplot(234)
        diagonal = np.array([self.charge_grid[i, i] for i in range(self.N)])
        positions = np.linspace(0, self.L * np.sqrt(2), self.N)
        ax4.plot(positions, diagonal, 'r-', linewidth=2)
        ax4.set_xlabel('Position diagonale (m)')
        ax4.set_ylabel('σ (C/m²)')
        ax4.set_title('Distribution le long de la diagonale')
        ax4.grid(True, alpha=0.3)
        
        # 5. Distribution des densités de charge
        ax5 = fig.add_subplot(235)
        ax5.hist(self.charge_grid.flatten(), bins=30, alpha=0.7, color='green', 
                edgecolor='black')
        ax5.set_xlabel('Densité de charge σ (C/m²)')
        ax5.set_ylabel('Nombre de patches')
        ax5.set_title('Distribution des densités de charge')
        ax5.grid(True, alpha=0.3)
        
        # 6. Effet de bord
        ax6 = fig.add_subplot(236)
        center_line = self.charge_grid[self.N//2, :]
        edge_line = self.charge_grid[0, :]
        x_pos = np.linspace(0, self.L, self.N)
        ax6.plot(x_pos, center_line, 'b-', linewidth=2, label='Centre (y=L/2)')
        ax6.plot(x_pos, edge_line, 'r-', linewidth=2, label='Bord (y=0)')
        ax6.set_xlabel('Position X (m)')
        ax6.set_ylabel('σ (C/m²)')
        ax6.set_title('Effet de bord: Centre vs Bord')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/visualisations_essentielles.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualisations essentielles generees!")
    
    def show_detailed_statistics(self):
        """Affiche des statistiques détaillées"""
        print("\n" + "=" * 70)
        print("STATISTIQUES DETAILLEES")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        charge_flat = self.charge_grid.flatten()
        print(f"\nDISTRIBUTION DE CHARGE:")
        print(f"   Minimum:      {np.min(charge_flat):.3e} C/m²")
        print(f"   Maximum:      {np.max(charge_flat):.3e} C/m²") 
        print(f"   Moyenne:      {np.mean(charge_flat):.3e} C/m²")
        print(f"   Ecart-type:   {np.std(charge_flat):.3e} C/m²")
        print(f"   Rapport max/min: {np.max(charge_flat)/np.min(charge_flat):.2f}")
        
        print(f"\nMATRICE Z:")
        print(f"   Dimensions:   {self.Z.shape}")
        print(f"   Condition:    {np.linalg.cond(self.Z):.2e}")
        
        total_charge = self.params.get('total_charge', np.sum(charge_flat) * (self.L/self.N)**2)
        capacitance = total_charge / 1.0
        print(f"\nCAPACITE:")
        print(f"   Charge totale: {total_charge:.3e} C")
        print(f"   Capacite:      {capacitance:.3e} F")
        
        potential_recalc = self.Z @ self.charge_coeffs
        error = np.abs(potential_recalc - 1.0)
        print(f"\nVERIFICATION:")
        print(f"   Erreur RMS:    {np.sqrt(np.mean(error**2)):.2e} V")
        print(f"   Erreur max:    {np.max(error):.2e} V")

    def run_analysis(self):
        """Lance l'analyse complète"""
        print("LANCEMENT DE L'ANALYSE")
        print(f"Dossier: {self.results_dir}")
        print("Reference: Gibson - Chapter 3, Section 3.1.2")
        
        self.load_all_data()
        self.show_data_summary()
        self.create_essential_visualizations()
        self.show_detailed_statistics()
        
        print("\nANALYSE TERMINEE AVEC SUCCES!")
        print("Tous les resultats sont sauvegardes dans:", self.results_dir)


class CompleteWireAnalysis:
    """
    ANALYSE COMPLÈTE DU FIL CHARGÉ
    Référence: Gibson - Chapter 3, Section 3.1.1
    """
    
    def __init__(self, results_dir="wire_200_segments"):
        self.results_dir = results_dir
        self.load_all_data()
        
    def load_all_data(self):
        """Charge TOUS les fichiers et explique leur contenu"""
        print("=" * 70)
        print("CHARGEMENT ET EXPLICATION DES FICHIERS")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        files_info = {
            'Z_matrix.npy': "MATRICE D'IMPÉDANCE - Interactions entre segments",
            'charge_coeffs.npy': "SOLUTION - Densités de charge λ en 1D", 
            'x_centers.npy': "POSITIONS - Coordonnées des centres des segments",
            'b_vector.npy': "VECTEUR EXCITATION - Potentiel imposé de 1V partout",
            'params.pkl': "PARAMÈTRES - Tous les paramètres de simulation"
        }
        
        self.data = {}
        self.explanations = {}
        
        for file, explanation in files_info.items():
            try:
                filepath = f"{self.results_dir}/{file}"
                if file.endswith('.npy'):
                    self.data[file] = np.load(filepath)
                elif file.endswith('.pkl'):
                    with open(filepath, 'rb') as f:
                        self.data[file] = pickle.load(f)
                
                self.explanations[file] = explanation
                print(f"{file:20} -> {explanation}")
                
            except Exception as e:
                print(f"{file:20} -> Erreur: {e}")
        
        self.Z = self.data.get('Z_matrix.npy')
        self.charge_coeffs = self.data.get('charge_coeffs.npy')
        self.x_centers = self.data.get('x_centers.npy')
        self.b = self.data.get('b_vector.npy')
        self.params = self.data.get('params.pkl')
        
        if self.params:
            self.N = self.params.get('N', 0)
            self.L = self.params.get('L', 1.0)
            self.a = self.params.get('a', 1e-3)
        
        print(f"Donnees chargees: {self.N} segments, L={self.L}m, a={self.a}m")
        print(f"Dossier d'analyse: {self.results_dir}")
    
    def show_data_summary(self):
        """Affiche un résumé complet de toutes les données"""
        print("\n" + "=" * 70)
        print("RESUME COMPLET DES DONNEES")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        for file, data in self.data.items():
            print(f"\n{file} : {self.explanations[file]}")
            
            if isinstance(data, np.ndarray):
                print(f"   Shape: {data.shape}, Type: {data.dtype}")
                print(f"   Min: {np.min(data):.2e}, Max: {np.max(data):.2e}, Mean: {np.mean(data):.2e}")
    
    def create_essential_visualizations(self):
        """Crée les visualisations essentielles pour le fil chargé"""
        print("\n" + "=" * 70)
        print("GENERATION DES VISUALISATIONS ESSENTIELLES")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        # Configuration pour 2x3 subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Distribution de charge le long du fil
        ax1 = fig.add_subplot(231)
        ax1.plot(self.x_centers, self.charge_coeffs, 'b-', linewidth=2)
        ax1.set_xlabel('Position le long du fil (m)')
        ax1.set_ylabel('Densité de charge (C/m)')
        ax1.set_title('Distribution de charge sur le fil')
        ax1.grid(True, alpha=0.3)
        
        # 2. Effet de bord - Comparaison début/milieu/fin
        ax2 = fig.add_subplot(232)
        start_idx = self.N // 10
        middle_idx = self.N // 2
        end_idx = 9 * self.N // 10
        
        positions = ['Début', 'Milieu', 'Fin']
        charges = [self.charge_coeffs[start_idx], self.charge_coeffs[middle_idx], self.charge_coeffs[end_idx]]
        
        bars = ax2.bar(positions, charges, color=['red', 'blue', 'red'])
        ax2.set_ylabel('Densité de charge (C/m)')
        ax2.set_title('Effet de bord: Début/Milieu/Fin')
        ax2.grid(True, alpha=0.3)
        
        # Ajout des valeurs sur les barres
        for bar, charge in zip(bars, charges):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height * 1.05, 
                    f'{charge:.2e}', ha='center', va='bottom')
        
        # 3. Matrice Z (log scale)
        ax3 = fig.add_subplot(233)
        display_size = min(20, self.N)
        im_z = ax3.imshow(np.log10(np.abs(self.Z[:display_size, :display_size])), cmap='viridis')
        ax3.set_xlabel('Index segment n')
        ax3.set_ylabel('Index segment m')
        ax3.set_title(f'Matrice Z (log|Zₘₙ| - {display_size}x{display_size})')
        plt.colorbar(im_z, ax=ax3, label='log₁₀|Zₘₙ|')
        
        # 4. Distribution le long du fil avec zoom sur les bords
        ax4 = fig.add_subplot(234)
        ax4.plot(self.x_centers, self.charge_coeffs, 'b-', linewidth=2)
        ax4.set_xlabel('Position le long du fil (m)')
        ax4.set_ylabel('Densité de charge (C/m)')
        ax4.set_title('Distribution de charge (vue détaillée)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Distribution des densités de charge
        ax5 = fig.add_subplot(235)
        ax5.hist(self.charge_coeffs, bins=30, alpha=0.7, color='green', 
                edgecolor='black')
        ax5.set_xlabel('Densité de charge (C/m)')
        ax5.set_ylabel('Nombre de segments')
        ax5.set_title('Distribution des densités de charge')
        ax5.grid(True, alpha=0.3)
        
        # 6. Erreur de résolution
        ax6 = fig.add_subplot(236)
        epsilon_0 = 8.854187817e-12
        potential_recalc = np.zeros(self.N)
        for m in range(self.N):
            for n in range(self.N):
                potential_recalc[m] += self.charge_coeffs[n] * self.Z[m, n]
        potential_recalc /= (4 * np.pi * epsilon_0)
        
        error = np.abs(potential_recalc - 1.0)
        ax6.plot(error, 'g-', alpha=0.7)
        ax6.set_xlabel('Index du segment')
        ax6.set_ylabel('Erreur (V)')
        ax6.set_title(f'Erreur de solution\nRMS: {np.sqrt(np.mean(error**2)):.2e} V')
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/visualisations_fil_essentielles.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualisations essentielles pour le fil generees!")
    
    def show_detailed_statistics(self):
        """Affiche des statistiques détaillées pour le fil"""
        print("\n" + "=" * 70)
        print("STATISTIQUES DETAILLEES")
        print(f"Dossier: {self.results_dir}")
        print("=" * 70)
        
        print(f"\nDISTRIBUTION DE CHARGE:")
        print(f"   Minimum:      {np.min(self.charge_coeffs):.3e} C/m")
        print(f"   Maximum:      {np.max(self.charge_coeffs):.3e} C/m") 
        print(f"   Moyenne:      {np.mean(self.charge_coeffs):.3e} C/m")
        print(f"   Ecart-type:   {np.std(self.charge_coeffs):.3e} C/m")
        print(f"   Rapport max/min: {np.max(self.charge_coeffs)/np.min(self.charge_coeffs):.2f}")
        
        print(f"\nMATRICE Z:")
        print(f"   Dimensions:   {self.Z.shape}")
        print(f"   Condition:    {np.linalg.cond(self.Z):.2e}")
        
        delta_x = self.L / self.N
        total_charge = np.sum(self.charge_coeffs * delta_x)
        capacitance = total_charge / 1.0
        print(f"\nCAPACITE:")
        print(f"   Charge totale: {total_charge:.3e} C")
        print(f"   Capacite:      {capacitance:.3e} F")
        
        # Vérification de la solution
        epsilon_0 = 8.854187817e-12
        potential_recalc = np.zeros(self.N)
        for m in range(self.N):
            for n in range(self.N):
                potential_recalc[m] += self.charge_coeffs[n] * self.Z[m, n]
        potential_recalc /= (4 * np.pi * epsilon_0)
        
        error = np.abs(potential_recalc - 1.0)
        print(f"\nVERIFICATION:")
        print(f"   Erreur RMS:    {np.sqrt(np.mean(error**2)):.2e} V")
        print(f"   Erreur max:    {np.max(error):.2e} V")

    def run_analysis(self):
        """Lance l'analyse complète pour le fil"""
        print("LANCEMENT DE L'ANALYSE DU FIL")
        print(f"Dossier: {self.results_dir}")
        print("Reference: Gibson - Chapter 3, Section 3.1.1")
        
        self.load_all_data()
        self.show_data_summary()
        self.create_essential_visualizations()
        self.show_detailed_statistics()
        
        print("\nANALYSE TERMINEE AVEC SUCCES!")
        print("Tous les resultats sont sauvegardes dans:", self.results_dir)


def analyze_specific_folder(folder_name):
    """Analyse un dossier spécifique (détecte automatiquement plaque ou fil)"""
    if not os.path.exists(folder_name):
        print(f"Le dossier '{folder_name}' n'existe pas!")
        available = [f for f in os.listdir() if os.path.isdir(f) and (f.startswith('plate_') or f.startswith('wire_'))]
        if available:
            print(f"Dossiers disponibles: {available}")
        return
    
    # Détection automatique du type de simulation
    if folder_name.startswith('plate_'):
        analyzer = CompletePlateAnalysis(folder_name)
    elif folder_name.startswith('wire_'):
        analyzer = CompleteWireAnalysis(folder_name)
    else:
        print(f"Type de simulation non reconnu pour le dossier: {folder_name}")
        return
    
    analyzer.run_analysis()


def analyze_all_simulations():
    """Analyse automatiquement tous les dossiers de simulation"""
    print("RECHERCHE DE TOUS LES DOSSIERS DE SIMULATION")
    print("=" * 70)
    
    # Trouve tous les dossiers de simulation
    simulation_folders = [f for f in os.listdir() if os.path.isdir(f) and (f.startswith('plate_') or f.startswith('wire_'))]
    
    if not simulation_folders:
        print("Aucun dossier de simulation trouve!")
        print("Executez d'abord les codes de simulation de la plaque ou du fil charge")
        return
    
    print(f"Dossiers trouves: {simulation_folders}")
    
    for folder in simulation_folders:
        print(f"\n" + "="*50)
        print(f"ANALYSE DU DOSSIER: {folder}")
        print("="*50)
        
        try:
            if folder.startswith('plate_'):
                analyzer = CompletePlateAnalysis(folder)
            else:
                analyzer = CompleteWireAnalysis(folder)
            analyzer.run_analysis()
        except Exception as e:
            print(f"Erreur avec {folder}: {e}")


if __name__ == "__main__":
    print("ANALYSE DES SIMULATIONS ELECTROMAGNETIQUES")
    print("=" * 70)
    
    # Option 1: Analyser un dossier spécifique
    #analyze_specific_folder("plate_40_patches")
    
    # Option 2: Analyser automatiquement tous les dossiers
    analyze_all_simulations()