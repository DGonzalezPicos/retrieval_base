import numpy as np
def select_species(line_species, species_wave, wmin, wmax):
    """
    Selects relevant species within a given wavelength range.
    
    Parameters:
    - line_species (list): List of species names.
    - species_wave (dict): Dictionary where keys are species and values are lists of wavelength ranges.
    - wmin (float): Minimum wavelength of the range.
    - wmax (float): Maximum wavelength of the range.
    
    Returns:
    - list: List of species with wavelength ranges overlapping with (wmin, wmax).
    """
    relevant_species = []
    for mol in line_species:
        if mol in species_wave:  # Ensure the species is in the dictionary
            for rmin, rmax in species_wave[mol]:
                # Check for overlap
                if wmin <= rmax and wmax >= rmin:
                    relevant_species.append(mol)
                    break  # No need to check further ranges for this molecule
    return relevant_species

# Example usage
line_species = ['12CO', 'CO2', 'H2O', 'SiO']
species_wave = {
    '12CO': [[1500, 1800], [2200, 2500], [4200, 5200]],
    'CO2': [[3900, 4600]],
    'SiO' : [[4900, 5200]],
}
# add line_species that are not in species_wave with (0, inf)
species_wave.update({s: [[0, np.inf]] for s in line_species if s not in species_wave})

wave_ranges = [
    [3000, 4000],
    [4000, 5000],
]

for wave_range in wave_ranges:
    wmin, wmax = wave_range
    result = select_species(line_species, species_wave, wmin, wmax)
    print(f"For wavelength range {wave_range}, relevant species are: {result}")
