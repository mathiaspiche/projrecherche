from minizinc import Instance, Model, Solver
import time
from datetime import timedelta
import re
import os
import glob

from torch_geometric.graphgym import train

#modelpath =
#filepath = 
#training_folder =
#val_folder = 
#solution_folder = 
def openDZN(path):
    with open(path, "r") as f:
        txt = f.read()
    m = re.search(r"nb_nodes\s*=\s*(\d+)", txt)
    if not m:
        raise RuntimeError(f"nb_nodes not found in {path}")
    N = int(m.group(1))

    mE = re.search(
        r"edges\s*=\s*array2d\s*\(\s*.*?\s*,\s*\[\s*(.*?)\s*\]\s*\)",
        txt,
        flags=re.S
    )
    if not mE:
        raise RuntimeError(f"edges array2d block not found in {path}")

    block = mE.group(1)

    nums = list(map(int, re.findall(r"\d+", block)))

    if len(nums) % 2 != 0:
        raise RuntimeError(
            f"Odd number of integers in edges array2d in {path}"
        )

    edges = [(nums[i] - 1, nums[i+1] - 1)
             for i in range(0, len(nums), 2)]

    return N, edges

def getSolutionDZN(
    solveur,
    filepath_dzn: str,
    modelpath: str = modelpath,
    out_dir: str | None = None,
    write: bool = True,
):

    model = Model(modelpath)
    model.add_file(filepath_dzn)
    if solveur == "Chuffed":
        solver = Solver.lookup("chuffed")
    elif solveur == "Gecode":
        solver = Solver.lookup("gecode")
    instance = Instance(solver, model)

    start = time.perf_counter()
    result = instance.solve(timeout=timedelta(milliseconds=400000))
    end = time.perf_counter()

    solve_time = end - start

    colors = result["couleurs"]
    colors_list = list(colors)
    stats = result.statistics
    props = stats.get("propagations", None)


    if write:
        if out_dir is None:
            out_dir = os.path.dirname(filepath_dzn)
        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(filepath_dzn))[0]
        out_path = os.path.join(out_dir, base + "_colors.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(" ".join(str(c) for c in colors_list))



    return colors_list, solve_time, props
