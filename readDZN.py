from minizinc import Instance, Model, Solver
import time
from datetime import timedelta
import re
import os
import glob

from torch_geometric.graphgym import train

modelpath = "C://Users//mathi//OneDrive//Documents//articlesIFT7020//projetRechercheIFT7020MiniZinc//solver.mzn"
filepath = "C://Users//mathi//OneDrive//Documents//articlesIFT7020//projetRechercheIFT7020MiniZinc//fichiers benchs//myciel4.dzn"
training_folder = "C:/Users/mathi/OneDrive/Documents/articlesIFT7020/projetRechercheIFT7020MiniZinc/test_set"
val_folder = "C:/Users/mathi/OneDrive/Documents/articlesIFT7020/projetRechercheIFT7020MiniZinc/test_set"
solution_folder = "C:/Users/mathi/OneDrive/Documents/articlesIFT7020/projetRechercheIFT7020MiniZinc/sols"
def openDZN(path):
    with open(path, "r") as f:
        txt = f.read()

    # ---------------------------
    # 1) Parse nb_nodes
    # ---------------------------
    m = re.search(r"nb_nodes\s*=\s*(\d+)", txt)
    if not m:
        raise RuntimeError(f"nb_nodes not found in {path}")
    N = int(m.group(1))

    # ---------------------------
    # 2) Parse array2d edges block
    # ---------------------------
    # Matches:  edges = array2d(..., [...]);
    mE = re.search(
        r"edges\s*=\s*array2d\s*\(\s*.*?\s*,\s*\[\s*(.*?)\s*\]\s*\)",
        txt,
        flags=re.S
    )
    if not mE:
        raise RuntimeError(f"edges array2d block not found in {path}")

    block = mE.group(1)

    # Extract ALL integers inside [...]:
    nums = list(map(int, re.findall(r"\d+", block)))

    # Check that count is even
    if len(nums) % 2 != 0:
        raise RuntimeError(
            f"Odd number of integers in edges array2d in {path}"
        )

    # Convert to zero-based edges
    edges = [(nums[i] - 1, nums[i+1] - 1)
             for i in range(0, len(nums), 2)]

    return N, edges

def getSolutionDZN(
    filepath_dzn: str,
    modelpath: str = modelpath,
    out_dir: str | None = None,
    write: bool = True,
):
    """
    Solve a .dzn instance and optionally write ONLY the colors to a file.

    File: <basename>_colors.txt in same folder (if out_dir is None)
    Content: one line, space-separated colors   e.g.  1 2 3 1 4 2 ...

    Returns:
        colors_list: Python list of colors
        solve_time:  float, wall-clock time in seconds
    """

    # Build and solve MiniZinc instance
    model = Model(modelpath)
    model.add_file(filepath_dzn)
    solver = Solver.lookup("chuffed")
    instance = Instance(solver, model)

    start = time.perf_counter()
    result = instance.solve(timeout=timedelta(milliseconds=60000))
    end = time.perf_counter()

    solve_time = end - start

    colors = result["couleurs"]
    colors_list = list(colors)  # plain Python list
    stats = result.statistics
    props = stats.get("propagations", None)


    if write:
        # Same directory as .dzn if out_dir is None
        if out_dir is None:
            out_dir = os.path.dirname(filepath_dzn)
        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(filepath_dzn))[0]
        out_path = os.path.join(out_dir, base + "_colors.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(" ".join(str(c) for c in colors_list))



    return colors_list, solve_time, props