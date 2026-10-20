"""Minimal PDB reading: every MODEL as a frame, with protein roles from atom names."""
import numpy as np

SYMBOLS = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}


def read(path):
    """positions (frames, atoms, 3), atomic numbers (atoms,), roles (atoms,) of ATOM records.

    Roles: 0 hydrogen, 1 side chain heavy atom, 2 backbone N C O, 3 CA.
    """
    frames, current, names, numbers = [], [], [], []
    with open(path) as file:
        for line in file:
            if line.startswith("ATOM"):
                current.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                if not frames:
                    name = line[12:16].strip()
                    element = line[76:78].strip() or name[0]
                    names.append(name)
                    numbers.append(SYMBOLS[element.capitalize()])
            elif line.startswith("ENDMDL") and current:
                frames.append(current)
                current = []
    if current:
        frames.append(current)
    numbers = np.asarray(numbers)
    roles = np.where(numbers == 1, 0, 1)
    roles[np.isin(names, ["N", "C", "O", "OXT"])] = 2
    roles[np.asarray(names) == "CA"] = 3
    return np.asarray(frames, dtype=np.float32), numbers, roles
