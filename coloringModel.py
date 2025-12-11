import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import solve

import readDZN
import time


modelpath = "C://Users//mathi//OneDrive//Documents//articlesIFT7020//projetRechercheIFT7020MiniZinc//solver.mzn"

gnn_path = r"C:\Users\mathi\OneDrive\Documents\articlesIFT7020\projetRechercheIFT7020MiniZinc\modele_entraine_GNN_10col.pt"

training_folder = r"C:\Users\mathi\OneDrive\Documents\articlesIFT7020\projetRechercheIFT7020MiniZinc\fichiers_benchs_3color"

nb_colors = 10
lambda_eff = 0.008
noise_scale = 0.001
def is_valid_coloring(colors, edges):
    """
    colors: 1D tensor of shape [N] containing integer color assignments
    edges: list of (u, v) pairs

    Returns:
        (valid, conflicts)
        valid: bool
        conflicts: list of (u, v) edges where colors match
    """
    colors = colors.detach().cpu()

    conflicts = [((u, v), (colors[u], colors[v])) for (u, v) in edges if colors[u] == colors[v]]
    valid = (len(conflicts) == 0)

    return valid, conflicts


def inject_high_confidence_constraints(
    solver_path: str,
    output_path: str,
    colors,
    confidence,
    conflicts,
    threshold: float = 0.95
):
    """
    NEW VERSION:
    Writes constraints of the form:

        constraint couleurs[i] <= predicted_color;

    Not couleurs[i] == predicted_color.

    This does NOT force the solver to use that color, it only restricts
    the domain to 1..pred_color.

    Constraints are injected only for:
        - nodes NOT in conflict
        - nodes with confidence >= threshold
    """

    # -------------------------------------------------------
    # 1. Extract conflict nodes robustly
    # -------------------------------------------------------
    conflict_nodes = set()

    for item in conflicts:

        # Case 1: ((u,v),(cu,cv))
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], (tuple, list))
            and len(item[0]) == 2
        ):
            u, v = item[0]

        # Case 2: (u,v)
        elif (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
        ):
            u, v = item

        else:
            raise ValueError(f"Unrecognized conflict format: {item}")

        conflict_nodes.add(int(u))
        conflict_nodes.add(int(v))

    print("Conflict nodes detected:", conflict_nodes)

    # -------------------------------------------------------
    # 2. Normalize tensor → python lists
    # -------------------------------------------------------
    if hasattr(colors, "detach"):
        colors = colors.detach().cpu().tolist()
    if hasattr(confidence, "detach"):
        confidence = confidence.detach().cpu().tolist()

    assert len(colors) == len(confidence), \
        "Color/confidence vector length mismatch"

    with open(solver_path, "r", encoding="utf-8") as f:
        model_text = f.read()

    # -------------------------------------------------------
    # 4. Build constraint block (using <= instead of ==)
    # -------------------------------------------------------
    lines = []
    lines.append("")
    lines.append("% --- Auto-generated high-confidence <= constraints ---")

    injected = 0

    for i, (col, conf_val) in enumerate(zip(colors, confidence)):

        if i in conflict_nodes:
            continue

        if float(conf_val) < threshold:
            continue

        # MiniZinc is 1-based
        idx_mzn = i + 1
        # convert predicted zero-based color to 1-based
        col_mzn = int(col) + 1

        # NEW: only upper-bound constraint
        lines.append(
            f"constraint couleurs[{idx_mzn}] <= {col_mzn};  % conf={conf_val:.4f}"
        )
        injected += 1

    if injected == 0:
        print("⚠ No constraints added: all nodes in conflict or below threshold.")

    # -------------------------------------------------------
    # 5. Write output file
    # -------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(model_text.rstrip() + "\n\n")
        f.write("\n".join(lines) + "\n")

    print(f"✅ Injected {injected} (<=) constraints into {output_path}")
    return output_path
class RUNCGNN(nn.Module):
    def __init__(self, k=128, nb_colors=nb_colors):
        super().__init__()

        self.k = k
        self.nb_colors = nb_colors

        # Shared initial state
        self.init_state = nn.Parameter(torch.randn(k) * 0.1)

        # Degree embedding (Permutation-invariant)
        self.deg_mlp = nn.Sequential(
            nn.Linear(1, k),
            nn.ReLU(),
            nn.Linear(k, k),
        )

        # Message MLP
        self.msg = nn.Sequential(
            nn.Linear(2 * k, 2 * k),
            nn.ReLU(),
            nn.Linear(2 * k, k)
        )

        # LSTM cell
        self.lstm = nn.LSTMCell(k, k)

        # LayerNorm for stability
        self.layer_norm = nn.LayerNorm(k)

        # Final projection to colors
        self.W = nn.Parameter(torch.randn(nb_colors, k) * 0.1)

    def forward(self, N, edges, max_steps, noise_scale=noise_scale, return_all=False):
        device = self.W.device

        # -----------------------------
        # Build edge index + degrees
        # -----------------------------
        if edges:
            u_idx = torch.tensor([u for (u, v) in edges], device=device)
            v_idx = torch.tensor([v for (u, v) in edges], device=device)

            deg = torch.zeros(N, device=device)
            deg.index_add_(0, u_idx, torch.ones_like(u_idx, dtype=torch.float))
            deg.index_add_(0, v_idx, torch.ones_like(v_idx, dtype=torch.float))
            deg = deg.clamp(min=1).unsqueeze(1)
        else:
            u_idx = torch.empty(0, dtype=torch.long, device=device)
            v_idx = torch.empty(0, dtype=torch.long, device=device)
            deg = torch.ones((N, 1), device=device)

        # -----------------------------
        # Initial embedding
        # -----------------------------
        s = self.init_state.unsqueeze(0).expand(N, -1)
        s = s + self.deg_mlp(deg)

        if noise_scale > 0:
            s = s + noise_scale * torch.randn_like(s)

        h = torch.zeros(N, self.k, device=device)

        phis = []  # ← store φ^(t) if return_all

        # -----------------------------
        # Recurrent message passing
        # -----------------------------
        for t in range(max_steps):
            msg_sum = torch.zeros(N, self.k, device=device)

            if u_idx.numel() > 0:
                su = s[u_idx]
                sv = s[v_idx]

                inp_uv = torch.cat([su, sv], dim=1)
                inp_vu = torch.cat([sv, su], dim=1)

                m_uv = self.msg(inp_uv)
                m_vu = self.msg(inp_vu)

                msg_sum.index_add_(0, u_idx, m_vu)
                msg_sum.index_add_(0, v_idx, m_uv)

            r = msg_sum / deg

            h, new_s = self.lstm(r, (h, s))
            s = self.layer_norm(s + new_s)

            if return_all:
                logits_t = (s @ self.W.t()) * 2.0
                phi_t = F.softmax(logits_t, dim=1)
                phis.append(phi_t)

        logits = (s @ self.W.t()) * 2.0

        if return_all:
            return logits, phis  # phis is list of length max_steps
        else:
            return logits

def coloring_loss(phi, edges, lambda_eff=lambda_eff, eps=1e-12):

    device = phi.device
    N, C = phi.shape

    if edges:
        u = torch.tensor([a for (a, b) in edges], device=device)
        v = torch.tensor([b for (a, b) in edges], device=device)

        same_color_prob = (phi[u] * phi[v]).sum(dim=1)
        loss_csp = -torch.log(1 - same_color_prob + eps).mean()
    else:
        loss_csp = torch.tensor(0.0, device=device)

    usage = phi.mean(dim=0)
    eff_colors = 1.0 / (usage.pow(2).sum() + eps)

    loss = loss_csp + lambda_eff * eff_colors

    return loss, loss_csp, eff_colors

"""
def curriculum_filter(graphs, epoch):
    if epoch < 10:
        maxN = 60
    elif epoch < 20:
        maxN = 100
    else :
        maxN = 999999
    return [(N, edges, name) for (N, edges, name) in graphs if N <= maxN]
"""

def load_training_graphs(folder):
    graphs = []
    for f in os.listdir(folder):
        if f.endswith(".dzn"):
            filepath = os.path.join(folder, f)
            try:
                N, edges = readDZN.openDZN(filepath)
                for u, v in edges:
                    if not (0 <= u < N and 0 <= v < N):
                        print(f"❌ BAD GRAPH: {filepath}")
                        print("N =", N)
                        print("Bad edge =", (u, v))
                        raise RuntimeError("Invalid edge found")
                graphs.append((N, edges, f))
            except Exception as e:
                print(f"Failed to load {f}: {e}")

    graphs.sort(key=lambda x: x[0])
    print(f"\nTotal training graphs: {len(graphs)}")
    return graphs


lambda_time = 0.95  # decay over time in loss

def train_model(
    epochs=200,
    folder=training_folder,
    nb_colors_train=nb_colors,
    lr=1e-4,
    batch_size=10,
    resume=True
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    graphs = load_training_graphs(folder)
    if not graphs:
        raise RuntimeError("No graphs found")

    gnn = RUNCGNN(nb_colors=nb_colors_train).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
    if resume and os.path.exists(gnn_path):
        print(f"🔄 Resuming training from checkpoint: {gnn_path}")
        state = torch.load(gnn_path, map_location=device, weights_only=True)

        # load model weights
        missing, unexpected = gnn.load_state_dict(state, strict=False)
        print("Loaded state_dict. Missing:", missing, "Unexpected:", unexpected)

        # load optimizer if present
        if "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
                print("✓ Optimizer state restored")
            except:
                print("⚠️ Optimizer state incompatible, resetting optimizer")

    else:
        print("🆕 Starting fresh training (no checkpoint loaded)")
    for epoch in range(1, epochs + 1):

        # small random walk on init state
        gnn.init_state.data += 0.0005 * torch.randn_like(gnn.init_state.data)


        random.shuffle(graphs)

        print(f"Training on {len(graphs)} graphs!")

        epoch_loss = epoch_csp = epoch_eff = 0.0
        num_batches = 0

        # -----------------------------
        # Iterate by batches of size 10
        # -----------------------------
        for idx in range(0, len(graphs), batch_size):

            gnn.train()
            graphs_batch = graphs[idx : idx + batch_size]

            batch_losses = []
            batch_csp = []
            batch_eff = []

            # accumulate loss for graphs in this batch
            for (N, edges, name) in graphs_batch:

                logits, phis = gnn.forward(
                    N, edges,
                    32,
                    noise_scale=noise_scale,
                    return_all=True
                )

                T = len(phis)  # number of time steps
                total_loss = 0.0
                w_sum = 0.0
                last_csp = last_eff = None

                for t, phi_t in enumerate(phis, start=1):
                    w_t = lambda_time ** (T - t)
                    loss_t, csp_t, eff_t = coloring_loss(phi_t, edges)

                    total_loss += w_t * loss_t
                    w_sum += w_t

                    last_csp = csp_t
                    last_eff = eff_t

                loss_graph = total_loss / w_sum
                batch_losses.append(loss_graph)
                batch_csp.append(last_csp)
                batch_eff.append(last_eff)

            # ----------------------------
            # Final batch loss = mean(loss)
            # ----------------------------
            loss = torch.stack(batch_losses).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), 1.0)
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            epoch_csp += torch.stack(batch_csp).mean().item()
            epoch_eff += torch.stack(batch_eff).mean().item()

            num_batches += 1
            print(idx)
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={epoch_loss/num_batches:.4f} "
            f"csp={epoch_csp/num_batches:.4f} "
            f"eff≈{epoch_eff/num_batches:.2f}"
        )
        torch.save(gnn.state_dict(), gnn_path)

def load_pretrained_model(nb_colors_model=nb_colors, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(gnn_path):
        raise FileNotFoundError(f"Pretrained model not found at: {gnn_path}")

    print(f"🔄 Loading pretrained weights from {gnn_path}")
    state = torch.load(gnn_path, map_location=device, weights_only=True)

    old_C, k = state["W"].shape
    if old_C != nb_colors_model:
        print(f"⚠️ Adjusting W from {old_C} to {nb_colors_model}")
        new_W = torch.randn(nb_colors_model, k) * 0.01
        if old_C <= nb_colors_model:
            new_W[:old_C] = state["W"]
        else:
            new_W = state["W"][:nb_colors_model]
        state["W"] = new_W

    gnn = RUNCGNN(nb_colors=nb_colors_model).to(device)
    gnn.load_state_dict(state, strict=False)
    gnn.eval()
    return gnn, device

def canonicalize_colors(colors: torch.Tensor) -> torch.Tensor:
    unique_vals, inverse = torch.unique(colors, sorted=True, return_inverse=True)
    return inverse
def solve_with_min_colors(
        dzn_path,
        nb_colors_model=nb_colors,
        max_model_colors=nb_colors,
        min_c=3,
        T_steps=100,        # number of recurrent steps in the GNN
        num_runs=25,        # how many random restarts
        gnn=None,
        device=None,
        noise_scale=0,      # small noise to randomize runs (optional)
):
    """
    RUN-CSP style decoder.

    Now returns:
        (best_colors, best_conf, new_model_path, elapsed_time)
    """

    start_time = time.perf_counter()

    # ---------------------------------------------------------
    # Load GNN and graph
    # ---------------------------------------------------------
    if gnn is None or device is None:
        gnn, device = load_pretrained_model(nb_colors_model)

    N, edges = readDZN.openDZN(dzn_path)
    max_c = min(max_model_colors, gnn.nb_colors)

    best_overall_colors = None
    best_overall_conflicts = float("inf")
    best_overall_c = None
    best_overall_conf = None   # NEW

    best_valid_colors = None
    best_valid_colors_used = float("inf")
    best_valid_c = None
    best_valid_conf = None     # NEW

    new_model_path = None

    with torch.no_grad():
        for c in range(min_c, max_c + 1):
            print(f"Trying with {c}")
            best_c_colors = None
            best_c_conflicts = float("inf")

            for run in range(num_runs):

                logits, phis = gnn.forward(
                    N,
                    edges,
                    T_steps,
                    noise_scale=noise_scale,
                    return_all=True
                )

                best_run_colors = None
                best_run_conflicts = float("inf")
                best_run_conf = None   # NEW

                for t, phi_t in enumerate(phis, start=1):

                    phi_c = phi_t[:, :c]

                    colors = phi_c.argmax(dim=1)
                    conf = phi_c.max(dim=1).values    # NEW: confidence

                    valid, conflicts = is_valid_coloring(colors, edges)
                    n_conf = len(conflicts)

                    # Track best run (even invalid)
                    if n_conf < best_run_conflicts:
                        best_run_conflicts = n_conf
                        best_run_colors = colors.clone()
                        best_run_conf = conf.clone()   # NEW

                    if valid:
                        can_col = canonicalize_colors(colors)
                        colors_used = can_col.unique().numel()
                        conf_valid = conf.clone()      # NEW

                        if (colors_used < best_valid_colors_used) or \
                           (colors_used == best_valid_colors_used and
                            (best_valid_c is None or c < best_valid_c)):
                            best_valid_colors = can_col.detach().cpu()
                            best_valid_colors_used = colors_used
                            best_valid_c = c
                            best_valid_conf = conf_valid.detach().cpu()  # NEW

                        break

                # End T-steps
                if best_run_conflicts < best_c_conflicts:
                    best_c_conflicts = best_run_conflicts
                    best_c_colors = best_run_colors.detach().cpu()
                    best_c_conf = best_run_conf.detach().cpu()   # NEW

            # End runs for this c
            if best_c_colors is not None and best_c_conflicts < best_overall_conflicts:
                best_overall_conflicts = best_c_conflicts
                best_overall_colors = best_c_colors
                best_overall_conf = best_c_conf      # NEW
                best_overall_c = c

    # ---------------------------------------------------------
    # Return best valid solution
    # ---------------------------------------------------------
    if best_valid_colors is not None:
        valid, conflicts = is_valid_coloring(best_valid_colors, edges)

        base_name = os.path.splitext(os.path.basename(dzn_path))[0]
        out_file = (
            f"C://Users//mathi//OneDrive//Documents//articlesIFT7020//"
            f"projetRechercheIFT7020MiniZinc//solver_warm_{base_name}.mzn"
        )

        elapsed = time.perf_counter() - start_time
        print("Confidence per node:", best_valid_conf)
        return best_valid_colors, best_valid_conf, out_file, elapsed  # NEW

    # ---------------------------------------------------------
    # No valid coloring: return best overall (invalid) attempt
    # ---------------------------------------------------------
    if best_overall_colors is not None:
        canonical_colors = canonicalize_colors(best_overall_colors)
        valid, conflicts = is_valid_coloring(canonical_colors, edges)

        out_file = (
            f"C://Users//mathi//OneDrive//Documents//articlesIFT7020//"
            f"projetRechercheIFT7020MiniZinc//solver{(best_overall_c, len(edges))}.mzn"
        )

        print("best_valid_colors =", best_valid_colors)
        print("best_overall_colors =", best_overall_colors)
        print("best overall confidence =", best_overall_conf)

        elapsed = time.perf_counter() - start_time
        print(conflicts)
        return canonical_colors, best_overall_conf, out_file, elapsed, conflicts   # NEW

    print("No best_overall_colors stored; returning None.")
    elapsed = time.perf_counter() - start_time
    return None, None, None, elapsed
def extract_conflict_nodes(conflicts):
    conflict_nodes = set()
    for item in conflicts:
        if isinstance(item, (tuple, list)):
            if len(item) == 2 and isinstance(item[0], int):
                u, v = item   # simple (u,v)
            else:
                # handles ((u,v),(cu,cv))
                u, v = item[0]
            conflict_nodes.add(int(u))
            conflict_nodes.add(int(v))
    return conflict_nodes
def getNewConstraints(colors, conflicts, max_fraction=0.05):
    """
    colors    : 1D list/array/tensor of int colors, indexed 0..N-1 (0-based)
    conflicts : list of either
                  - (u, v) pairs, or
                  - ((u, v), (color_u, color_v)) tuples
    max_fraction : maximum fraction of nodes to fix as hard equality constraints

    Returns:
        list of MiniZinc constraint strings of the form:
        'constraint couleurs[i] == c;'
        where i is 1-based and c is the (1-based) color.
    """

    # --- 0) Normalize conflicts to a list of (u, v) ints ---
    norm_conflicts = []
    for item in conflicts:
        # case 1: plain edge (u, v)
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
        ):
            u, v = item

        # case 2: ((u, v), (color_u, color_v)) or similar
        elif (
            isinstance(item, (tuple, list))
            and len(item) >= 1
            and isinstance(item[0], (tuple, list))
            and len(item[0]) == 2
        ):
            u, v = item[0]

        else:
            raise ValueError(f"Unrecognized conflict format: {item!r}")

        norm_conflicts.append((int(u), int(v)))

    # Convert colors to a plain list if needed (handles tensors)
    try:
        colors_list = list(colors)
    except TypeError:
        # e.g. PyTorch tensor
        colors_list = colors.detach().cpu().tolist()

    n = len(colors_list)
    if n == 0:
        return []

    # 1) Collect all nodes that appear in any conflict
    conflict_nodes = set()
    for u, v in norm_conflicts:
        conflict_nodes.add(u)
        conflict_nodes.add(v)

    # 2) Nodes that are never in a conflict
    non_conflict_nodes = [i for i in range(n) if i not in conflict_nodes]

    if len(non_conflict_nodes) == 0:
        return []

    # 3) Choose at most max_fraction of all nodes to fix
    max_nodes = int(max_fraction * n)
    if max_nodes < 1:
        max_nodes = 1  # always allow at least one node to be fixed

    if len(non_conflict_nodes) > max_nodes:
        # sample them regularly across the list for coverage
        step = max(1, len(non_conflict_nodes) // max_nodes)
        selected_nodes = non_conflict_nodes[::step][:max_nodes]
    else:
        selected_nodes = non_conflict_nodes

    # 4) Build equality constraints for selected nodes
    constraints = []
    for i in selected_nodes:
        color0 = int(colors_list[i])      # 0-based color from GNN
        color1 = color0 + 1               # 1-based for MiniZinc (adjust if needed)
        mi = i + 1                        # 1-based index for couleurs[]

        constraints.append(
            f"constraint couleurs[{mi}] == {color1};"
        )

    return constraints


def writeConstraints(
    solver_path,
    colors,
    conflicts,
    out_path=None,
    header_comment=None,
    max_fraction=1,
):
    """
    Creates a *new* MiniZinc model file that contains:
      - the original contents of `solver_path`
      - plus auto-generated constraints at the end, fixing up to
        `max_fraction` of all nodes to the given colors (for non-conflict nodes).

    solver_path   : path to the original .mzn file (kept unchanged)
    colors        : same as for getNewConstraints
    conflicts     : same as for getNewConstraints
    out_path      : path of the NEW .mzn file to write.
                    If None, we auto-generate "<solver_path>_with_constraints.mzn"
    header_comment: optional string comment to write before constraints
    max_fraction  : maximum fraction of nodes allowed to appear in constraints

    Returns:
        out_path, constraints
    """
    constraints = getNewConstraints(colors, conflicts, max_fraction=max_fraction)

    if not constraints:
        return None, constraints

    if header_comment is None:
        header_comment = "% Auto-generated equality constraints from GNN\n"

    # Decide output file name
    if out_path is None:
        root, ext = os.path.splitext(solver_path)
        if ext == "":
            ext = ".mzn"
        out_path = f"{root}_with_constraints{ext}"

    # Read original model
    with open(solver_path, "r", encoding="utf-8") as f:
        original_model = f.read()

    # Write new file: original + constraints
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(original_model.rstrip() + "\n\n")
        f.write(header_comment)
        for c in constraints:
            f.write(c + "\n")

    return out_path, constraints
