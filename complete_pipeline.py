import os
import glob
import matplotlib.pyplot as plt


from readDZN import (
    getSolutionDZN,
    openDZN,
)
from coloringModel import(
    solve_with_min_colors,
    inject_high_confidence_constraints,
)

original_solver = "C://Users//mathi//OneDrive//Documents//articlesIFT7020//projetRechercheIFT7020MiniZinc//solver.mzn"
output_solver = r"C:\Users\mathi\OneDrive\Documents\articlesIFT7020\projetRechercheIFT7020MiniZinc\solver_conf.mzn"
dzn_folder = r"C:\Users\mathi\OneDrive\Documents\articlesIFT7020\projetRechercheIFT7020MiniZinc\testofficiel"
def run_full_pipeline(dzn_folder):
    results = []

    dzn_files = sorted(glob.glob(dzn_folder + "/*.dzn"))
    print(f"Found {len(dzn_files)} instances")

    for path in dzn_files:
        print("\n===============================")
        print("Processing:", os.path.basename(path))
        print("===============================")

        # -----------------------------------
        # 1) GNN RUN + inject constraints
        # -----------------------------------
        (
            colors,
            confidence,
            solver_file_out,
            gnn_time,
            *rest
        ) = solve_with_min_colors(path)

        # If invalid: fallback returns conflicts in 'rest'
        conflicts = rest[-1] if len(rest) > 0 else []

        # Create the new solver file by injecting constraints
        inject_high_confidence_constraints(
            solver_path=original_solver,
            output_path=output_solver,
            colors=colors,
            conflicts=conflicts,
            confidence=confidence,
            threshold=0.95
        )

        # Extract N
        N, _ = openDZN(path)

        # -----------------------------------
        # 2) SOLVE WITH ORIGINAL SOLVER
        # -----------------------------------
        print("Solving with original solver...")
        base_colors, base_time, base_props = getSolutionDZN(
            filepath_dzn=path,
            modelpath=original_solver,
            write=False
        )

        # -----------------------------------
        # 3) SOLVE WITH INJECTED CONSTRAINTS
        # -----------------------------------
        print("Solving with injected-constraint solver...")
        new_colors, new_time, new_props = getSolutionDZN(
            filepath_dzn=path,
            modelpath=output_solver,
            write=False
        )

        # Save results
        results.append({
            "name": os.path.basename(path),
            "N": N,
            "base_time": base_time,
            "new_time": new_time,
            "base_props": base_props,
            "new_props": new_props
        })

        print(f"Done {path}: baseline={base_time:.3f}s new={new_time:.3f}s")

    return results


# ============================================
# PLOTTING FUNCTION
# ============================================
def plot_results(results):
    Ns = [r["N"] for r in results]
    base_t = [r["base_time"] for r in results]
    new_t = [r["new_time"] for r in results]
    base_p = [r["base_props"] for r in results]
    new_p = [r["new_props"] for r in results]

    # ---- TIME PLOT ------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(Ns, base_t, marker="o", label="Original solver time")
    plt.plot(Ns, new_t, marker="o", label="Solver with injected constraints")
    plt.xlabel("N (number of nodes)")
    plt.ylabel("Solve time (seconds)")
    plt.title("Solver Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- PROPAGATION PLOT -----------------
    plt.figure(figsize=(10, 5))
    plt.plot(Ns, base_p, marker="o", label="Original propagations")
    plt.plot(Ns, new_p, marker="o", label="New propagations")
    plt.xlabel("N (number of nodes)")
    plt.ylabel("Propagations")
    plt.title("Solver Propagation Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

run_full_pipeline(dzn_folder)