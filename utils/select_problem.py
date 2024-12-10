# Mapping problem names to functions
from heuristic.problems.MKP import MKP1, MKP2, MKP3, MKP4, MKP5, MKP6, MKP7, MKP8, MKP9, MKP10

def select_problem():
    problems = {
        "MKP1": MKP1,
        "MKP2": MKP2,
        "MKP3": MKP3,
        "MKP4": MKP4,
        "MKP5": MKP5,
        "MKP6": MKP6,
        "MKP7": MKP7,
        "MKP8": MKP8,
        "MKP9": MKP9,
        "MKP10": MKP10,
    }

    Probleme = input(f"Donner le nom de problème ({', '.join(problems.keys())}): ").strip()

    if Probleme not in problems:
        print("Ce problème n'existe pas. Veuillez réessayer.")
        return None, None

    func = problems[Probleme]

    D = 28 if Probleme in ["MKP1", "MKP2", "MKP3", "MKP4", "MKP5", "MKP6"] else (105 if Probleme in ["MKP7", "MKP8"] else 60)

    return func, D
