
import os
import math
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import cvxpy as cp

ROOT = os.getcwd()
MODEL_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
META_PATH = os.path.join(MODEL_DIR, "meta.pkl")
MODEL_PTH = os.path.join(MODEL_DIR, "lstm_best.pth")
RESID_PATH = os.path.join(MODEL_DIR, "residuals_val.npy")
RETURNS_CSV = os.path.join(DATA_DIR, "returns.csv")


M_SYNTH = 2000       
CLAMP = 0.2          
DEVICE = "cpu"

class MultiLSTM(nn.Module):
    def __init__(self, n_assets, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=n_assets, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, n_assets)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def load_artifacts():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"meta.pkl not found at {META_PATH}")
    meta = joblib.load(META_PATH)
    tickers = meta.get("tickers")
    mu = np.array(meta.get("mu"))
    sigma = np.array(meta.get("sigma"))
    seq_len = int(meta.get("seq_len", 20))

    if not os.path.exists(RETURNS_CSV):
        raise FileNotFoundError(f"returns.csv not found at {RETURNS_CSV}")
    returns_df = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True).sort_index()

    if not os.path.exists(RESID_PATH):
        raise FileNotFoundError(f"residuals_val.npy not found at {RESID_PATH}")
    residuals = np.load(RESID_PATH)

    if not os.path.exists(MODEL_PTH):
        raise FileNotFoundError(f"model weights not found at {MODEL_PTH}")
    state = torch.load(MODEL_PTH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        state_dict = state
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    ih_keys = [k for k in state_dict.keys() if k.startswith("lstm.weight_ih_l")]
    if len(ih_keys) == 0:
        fc_key = next((k for k in state_dict.keys() if k.endswith("fc.weight") or k == "fc.weight"), None)
        if fc_key is None:
            hidden_size = meta.get("hidden_size", 128)
            num_layers = meta.get("num_layers", 2)
        else:
            fc_w = state_dict[fc_key]
            hidden_size = int(fc_w.shape[1])
            num_layers = meta.get("num_layers", 2)
    else:
        wih0 = state_dict[ih_keys[0]]
        hidden_size = int(wih0.shape[0] // 4)
        layer_idxs = [int(k.split("l")[-1].split('.')[0]) for k in ih_keys]
        num_layers = max(layer_idxs) + 1

    n_assets = len(tickers)
    model = MultiLSTM(n_assets, hidden_size=hidden_size, num_layers=num_layers)
    try:
        model.load_state_dict(state_dict)
    except Exception:
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            model.load_state_dict(state["state_dict"])
        else:
            raise
    model.to(DEVICE)
    model.eval()

    return {
        "meta": meta,
        "tickers": tickers,
        "mu": mu,
        "sigma": sigma,
        "seq_len": seq_len,
        "returns_df": returns_df,
        "residuals": residuals,
        "model": model
    }

ART = load_artifacts()
def generate_paths_bootstrap(model, seed_window_orig, residuals_val, M=M_SYNTH, H=21, seq_len=None, clamp=CLAMP, device=DEVICE, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = seed_window_orig.shape[1]
    C = np.zeros((M, n), dtype=np.float64)
    mu = ART["mu"]
    sigma = ART["sigma"]
    for m in range(M):
        win = seed_window_orig.copy().astype(np.float64)
        cum = np.ones(n, dtype=np.float64)
        for h in range(H):
            inp_std = ((win - mu) / sigma).astype(np.float32)[None, :, :]  # (1, seq_len, n)
            with torch.no_grad():
                t_inp = torch.from_numpy(inp_std).to(device)
                pred_std = model(t_inp).cpu().numpy().reshape(-1)
            pred_orig = pred_std * sigma + mu
            idx = rng.integers(0, residuals_val.shape[0])
            eta = residuals_val[idx]
            sampled = pred_orig + eta
            sampled = np.clip(sampled, -clamp, clamp)
            cum *= (1.0 + sampled)
            win = np.vstack([win[1:, :], sampled])
        C[m, :] = cum - 1.0
    return C

def min_cvar(C, alpha=0.95):
    M, n = C.shape
    w = cp.Variable(n)
    t = cp.Variable(1)
    u = cp.Variable(M, nonneg=True)
    R = C @ w
    constraints = [u >= -R - t, cp.sum(w) == 1, w >= 0]
    cvar_expr = t + (1.0/(1 - alpha)) * (cp.sum(u) / M)
    prob = cp.Problem(cp.Minimize(cvar_expr), constraints)
    for solver in (cp.ECOS, cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return float(cvar_expr.value), np.array(w.value).reshape(-1)
        except Exception:
            continue
    return float("inf"), None

def solve_cvar_maxret(C, alpha=0.95, loss_limit=0.05):
    M, n = C.shape
    w = cp.Variable(n)
    t = cp.Variable(1)
    u = cp.Variable(M, nonneg=True)
    R = C @ w
    cvar_expr = t + (1.0/(1 - alpha)) * (cp.sum(u) / M)
    constraints = [u >= -R - t, cvar_expr <= loss_limit, cp.sum(w) == 1, w >= 0]
    obj = cp.Maximize(cp.sum(R) / M)
    prob = cp.Problem(obj, constraints)
    for solver in (cp.ECOS, cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                w_opt = np.array(w.value).reshape(-1)
                pred_exp = float((C @ w_opt).mean())
                pred_cvar = float(cvar_expr.value)
                return w_opt, {"predicted_exp": pred_exp, "predicted_cvar": pred_cvar, "solver": str(solver), "status": prob.status}
        except Exception:
            continue
    raise RuntimeError("CVaR optimization infeasible or solver failure")

def weights_to_dict(tickers, w):
    return {t: float(wi) for t, wi in zip(tickers, w)}

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<title>CVaR-LSTM Portfolio Predictor</title>
<h2>CVaR-LSTM Portfolio Predictor</h2>
<form action="/optimize" method="post">
  Horizon (H days): <input type="number" name="horizon" value="21" min="1" required><br>
  Loss limit (e.g. 0.05 for 5%): <input type="text" name="loss_limit" value="0.05" required><br>
  Alpha (CVaR confidence, e.g. 0.95): <input type="text" name="alpha" value="0.95" required><br>
  <input type="submit" value="Compute expected prediction">
</form>
<hr>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/optimize", methods=["POST"])
def optimize_form():
    try:
        horizon = int(request.form.get("horizon", 21))
        loss_limit = float(request.form.get("loss_limit", 0.05))
        alpha = float(request.form.get("alpha", 0.95))
    except Exception as e:
        return render_template_string("<p>Bad input: {{err}}</p>", err=str(e)), 400

    try:
        result = compute_prediction(horizon=horizon, loss_limit=loss_limit, alpha=alpha)
    except Exception as e:
        return render_template_string("<p>Error: {{err}}</p>", err=str(e)), 500

    html = ["<h3>Result</h3>"]
    if not result["feasible"]:
        html.append(f"<p><b>Infeasible:</b> the requested loss limit {loss_limit:.6f} is below the minimal achievable CVaR ({result['min_cvar']:.6f}) for the current market seed.</p>")
        if result.get("w_min") is not None:
            html.append("<p>Suggested minimal-loss portfolio (achieves min-CVaR):</p>")
            html.append("<table border='1'><tr><th>Ticker</th><th>Weight</th></tr>")
            for t, wv in result["w_min"].items():
                html.append(f"<tr><td>{t}</td><td>{wv:.6f}</td></tr>")
            html.append("</table>")
        html.append("<p>Suggestion: choose loss >= minimal CVaR above or adjust policy (allow shorts / increase loss limit).</p>")
    else:
        exp_pct = result.get("expected_profit_pct", None)
        html.append(f"<p>Predicted expected H-day portfolio return: <b>{result['predicted_expected']:.6f}</b> (fraction)</p>")
        if exp_pct is not None:
            html.append(f"<p><b>Predicted expected profit percentage:</b> {exp_pct:.3f}%</p>")
        html.append("<h4>Weights</h4><table border='1'><tr><th>Ticker</th><th>Weight</th></tr>")
        for t, wv in result["weights"].items():
            html.append(f"<tr><td>{t}</td><td>{wv:.6f}</td></tr>")
        html.append("</table>")
    html.append("<p><a href='/'>Back</a></p>")
    return render_template_string("\n".join(html))

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.get_json(force=True)
    try:
        horizon = int(data.get("horizon", 21))
        loss_limit = float(data.get("loss_limit", 0.05))
        alpha = float(data.get("alpha", 0.95))
    except Exception as e:
        return jsonify({"error": "bad input", "detail": str(e)}), 400

    try:
        result = compute_prediction(horizon=horizon, loss_limit=loss_limit, alpha=alpha)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

def compute_prediction(horizon=21, loss_limit=0.05, alpha=0.95):
    tickers = ART["tickers"]
    seq_len = int(ART["seq_len"])
    returns_df = ART["returns_df"]
    residuals = ART["residuals"]
    model = ART["model"]

    if len(returns_df) < seq_len + 1:
        raise RuntimeError("Not enough data to form a seed window.")

    seed_window = returns_df.values[-seq_len:, :]

    C = generate_paths_bootstrap(model, seed_window, residuals, M=M_SYNTH, H=horizon, seq_len=seq_len, clamp=CLAMP, device=DEVICE, rng_seed=42)

    mincv, w_min = min_cvar(C, alpha=alpha)
    if not np.isfinite(mincv):
        return {"feasible": False, "min_cvar": None, "error": "Could not compute min-CVaR numerically", "horizon": horizon, "alpha": alpha, "loss_limit": loss_limit}

    if mincv > loss_limit + 1e-12:
        return {
            "feasible": False,
            "min_cvar": float(mincv),
            "w_min": weights_to_dict(tickers, w_min) if w_min is not None else None,
            "horizon": horizon,
            "alpha": alpha,
            "loss_limit": loss_limit
        }

    w_opt, info = solve_cvar_maxret(C, alpha=alpha, loss_limit=loss_limit)

    predicted_expected = float(info["predicted_exp"])
    predicted_cvar = float(info["predicted_cvar"])
    expected_profit_pct = predicted_expected * 100.0

    return {
        "feasible": True,
        "min_cvar": float(mincv),
        "predicted_expected": predicted_expected,
        "expected_profit_pct": expected_profit_pct,
        "predicted_cvar": predicted_cvar,
        "weights": weights_to_dict(tickers, w_opt),
        "horizon": horizon,
        "alpha": alpha,
        "loss_limit": loss_limit,
        "used_M": M_SYNTH,
        "solver": info.get("solver", None),
        "status": info.get("status", None)
    }

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
