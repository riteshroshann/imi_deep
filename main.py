"""main.py — Improved 9-Stage CFRP Pipeline"""
import argparse, logging, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("CFRP_Pipeline")
SEED = 42
np.random.seed(SEED)

def _dirs():
    for d in ["data/raw","data/processed","results/figures","results/tables","logs"]:
        os.makedirs(d, exist_ok=True)

def _torch_seed():
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def _save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(metrics).T.to_csv(path)
    logger.info("Saved → %s", path)

# ── Stage 1: Data ─────────────────────────────────────────────────────────────
def stage_load_data(data_path, force_raw=False):
    from src.data_loader import download_dataset, parse_nasa_composites, normalize_signals, create_splits
    from src.feature_extraction import build_feature_matrix, reduce_dimensions
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Loading & Feature Engineering")
    logger.info("=" * 60)
    raw_dir = download_dataset(data_path, force_raw=force_raw)
    dataset = parse_nasa_composites(raw_dir, max_samples_per_specimen=200, seed=SEED, force_raw=force_raw, signal_length=2000)
    signals_norm, norm_params = normalize_signals(dataset["signals"])
    dataset["signals"] = signals_norm
    sig_shape = dataset["signals"].shape
    is_tabular = sig_shape[-1] <= 64
    logger.info("Signal shape: %s | Tabular: %s", sig_shape, is_tabular)
    features, feat_names = build_feature_matrix(dataset)
    # Damage Index
    signals = dataset["signals"]
    if is_tabular:
        n_base = max(1, int(0.10 * len(signals)))
        baseline = signals[:n_base].mean(axis=0, keepdims=True)
        diff = signals - baseline
        di_scalar = np.linalg.norm(diff.reshape(len(signals), -1), axis=1, keepdims=True)
        di = np.tile(di_scalar, (1, sig_shape[1]))
    else:
        from src.feature_extraction import compute_damage_index
        di = compute_damage_index(signals)
    # UMAP
    try:
        umap_emb = reduce_dimensions(features, method="umap", seed=SEED)
    except Exception:
        umap_emb = reduce_dimensions(features, method="pca", seed=SEED)
    splits = create_splits(dataset, seed=SEED)
    np.savez_compressed("data/processed/features.npz", features=features, di=di, umap=umap_emb)
    logger.info("Stage 1 done: %d samples, %d features", features.shape[0], features.shape[1])
    return {"dataset": dataset, "features": features, "feature_names": feat_names,
            "damage_index": di, "umap_embeddings": umap_emb, "splits": splits,
            "norm_params": norm_params, "is_tabular": is_tabular}

# ── Stage 2: Baselines ────────────────────────────────────────────────────────
def stage_baselines(processed):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from src.uncertainty import comprehensive_rul_metrics
    logger.info("=" * 60); logger.info("STAGE 2: Baseline Models"); logger.info("=" * 60)
    features, dataset = processed["features"], processed["dataset"]
    rng = np.random.default_rng(SEED)
    n = len(dataset["rul"])
    perm = rng.permutation(n)
    n_train, n_val = int(0.6*n), int(0.2*n)
    tr, va, te = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(features[tr]); X_va = scaler.transform(features[va]); X_te = scaler.transform(features[te])
    y_tr, y_va, y_te = dataset["rul"][tr], dataset["rul"][va], dataset["rul"][te]
    results, models_trained, preds = {}, {}, {}
    # LinReg
    lr = LinearRegression().fit(X_tr, y_tr)
    p_lr = lr.predict(X_te)
    results["LinReg"] = comprehensive_rul_metrics(p_lr, y_te); models_trained["LinReg"] = lr
    preds["LinReg"] = {"y_true": y_te, "y_pred": p_lr, "y_std": None}
    logger.info("  LinReg  RMSE=%.4f R²=%.3f", results["LinReg"]["rmse"], results["LinReg"]["r2"])
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    p_rf = rf.predict(X_te)
    results["RF"] = comprehensive_rul_metrics(p_rf, y_te); models_trained["RF"] = rf
    preds["RF"] = {"y_true": y_te, "y_pred": p_rf, "y_std": None}
    logger.info("  RF      RMSE=%.4f R²=%.3f", results["RF"]["rmse"], results["RF"]["r2"])
    # XGBoost
    try:
        import xgboost as xgb
        xm = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
        xm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_xgb = xm.predict(X_te)
        results["XGBoost"] = comprehensive_rul_metrics(p_xgb, y_te); models_trained["XGBoost"] = xm
        preds["XGBoost"] = {"y_true": y_te, "y_pred": p_xgb, "y_std": None}
        logger.info("  XGBoost RMSE=%.4f R²=%.3f", results["XGBoost"]["rmse"], results["XGBoost"]["r2"])
    except ImportError:
        logger.warning("XGBoost not available")
    _save_metrics(results, "results/tables/baseline_metrics.csv")
    return {"results": results, "models": models_trained, "scaler": scaler,
            "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te, "predictions": preds}

# ── Stage 3: Deep Learning ────────────────────────────────────────────────────
def _train_dl(model, train_loader, val_loader, device, epochs=80, lr=1e-3):
    import torch, torch.nn as nn
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    criterion = nn.MSELoss()
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        vl = 0; nv = 0
        import torch
        with torch.no_grad():
            for xb, yb in val_loader:
                vl += criterion(model(xb.to(device)), yb.to(device)).item(); nv += 1
        avg_vl = vl / max(nv, 1)
        if avg_vl < best_val:
            best_val = avg_vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 20 == 0:
            logger.info("  Epoch %3d | val=%.4f", epoch+1, avg_vl)
    if best_state: model.load_state_dict(best_state)
    return model

def _eval_dl(model, loader, device):
    import torch
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for xb, yb in loader:
            all_p.append(model(xb.to(device)).cpu().numpy())
            all_t.append(yb.numpy())
    return np.concatenate(all_p), np.concatenate(all_t)

def stage_deep_learning(processed, epochs=80):
    import torch
    from src.data_loader import create_splits, create_dataloaders
    from src.models import CNN1D, BiLSTMAttention, SensorTransformer, SpatialTemporalAttention
    from src.models.tcn import TemporalConvNet
    from src.uncertainty import comprehensive_rul_metrics
    logger.info("=" * 60); logger.info("STAGE 3: Deep Learning Models"); logger.info("=" * 60)
    device = _torch_seed()
    dataset = processed["dataset"]
    splits = create_splits(dataset, seed=SEED)
    loaders = create_dataloaders(splits, task="rul", batch_size=32)
    nc = dataset["signals"].shape[1]; sl = dataset["signals"].shape[2]
    logger.info("DL input: (batch, %d, %d)", nc, sl)
    configs = [
        ("HybridSTA",   SpatialTemporalAttention(n_sensors=nc, signal_length=sl, task="rul", dropout=0.2)),
    ]
    if sl <= 64:
        configs = [
            ("CNN1D",       CNN1D(n_sensors=nc, signal_length=sl, task="rul", dropout=0.2)),
            ("BiLSTM",      BiLSTMAttention(n_sensors=nc, signal_length=sl, task="rul", dropout=0.3)),
            ("TCN",         TemporalConvNet(n_sensors=nc, signal_length=sl, task="rul", dropout=0.2)),
            ("Transformer", SensorTransformer(n_sensors=nc, signal_length=sl, task="rul", dropout=0.1)),
        ] + configs
    dl_results, dl_models, dl_preds = {}, {}, {}
    for name, model in configs:
        logger.info("Training %s (%d params)...", name, model.num_parameters)
        t0 = time.time()
        model = _train_dl(model, loaders["train"], loaders["val"], device, epochs=epochs)
        y_pred, y_true = _eval_dl(model, loaders["test"], device)
        m = comprehensive_rul_metrics(y_pred, y_true)
        m["train_time_s"] = time.time() - t0; m["n_params"] = model.num_parameters
        dl_results[name] = m; dl_models[name] = model
        dl_preds[name] = {"y_true": y_true, "y_pred": y_pred, "y_std": None}
        logger.info("  %s RMSE=%.4f R²=%.3f NASA=%.2f (%.0fs)",
                    name, m["rmse"], m["r2"], m["nasa_score"], m["train_time_s"])
    _save_metrics(dl_results, "results/tables/deep_learning_metrics.csv")
    return {"results": dl_results, "models": dl_models, "predictions": dl_preds, "loaders": loaders}

# ── Stage 4: PINN ─────────────────────────────────────────────────────────────
def stage_pinn(processed):
    import torch
    from src.models.pinn import PhysicsInformedNet, PINNTrainer
    logger.info("=" * 60); logger.info("STAGE 4: PINN"); logger.info("=" * 60)
    device = _torch_seed(); dataset = processed["dataset"]
    di = processed["damage_index"]; life_frac = 1.0 - dataset["rul"]
    layup_oh = np.zeros((len(dataset["layup_id"]), 3), dtype=np.float32)
    for i, lid in enumerate(dataset["layup_id"]): layup_oh[i, int(lid)] = 1.0
    di_in = di[:, :16] if di.shape[1] >= 16 else di
    X_pinn = np.hstack([life_frac.reshape(-1,1), di_in, layup_oh]).astype(np.float32)
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(X_pinn))
    n_tr = int(0.8 * len(X_pinn)); tr, te = perm[:n_tr], perm[n_tr:]
    X_tr = torch.FloatTensor(X_pinn[tr]).to(device)
    stiff_tr = torch.FloatTensor(dataset["stiffness_ratio"][tr]).to(device)
    str_tr   = torch.FloatTensor(dataset["strength_ratio"][tr]).to(device)
    cyc_tr   = torch.FloatTensor(life_frac[tr]).to(device)
    model = PhysicsInformedNet(input_dim=X_pinn.shape[1], hidden_dim=128, n_blocks=4, lambda_physics=0.1).to(device)
    trainer = PINNTrainer(model, lr=1e-3, lambda_schedule="cosine_restart", max_epochs=300, batch_size=256)
    train_data = {"x": X_tr, "stiffness": stiff_tr, "strength": str_tr, "cycles": cyc_tr}
    for epoch in range(300):
        losses = trainer.train_epoch(train_data, epoch)
        if (epoch+1) % 50 == 0:
            logger.info("  Epoch %3d | total=%.4f data=%.4f physics=%.4f",
                        epoch+1, losses["total"], losses["data"], losses["physics"])
    model.eval()
    with torch.no_grad():
        sp, stp = model(torch.FloatTensor(X_pinn[te]).to(device))
    from sklearn.metrics import mean_squared_error
    sr = float(np.sqrt(mean_squared_error(dataset["stiffness_ratio"][te], sp.cpu().numpy())))
    st = float(np.sqrt(mean_squared_error(dataset["strength_ratio"][te],  stp.cpu().numpy())))
    logger.info("  PINN Stiffness RMSE=%.4f | Strength RMSE=%.4f", sr, st)
    return {"model": model, "trainer": trainer, "history": trainer.history,
            "stiffness_pred": sp.cpu().numpy(), "stiffness_true": dataset["stiffness_ratio"][te],
            "strength_pred":  stp.cpu().numpy(), "strength_true": dataset["strength_ratio"][te],
            "cycles_norm": life_frac[te]}

# ── Stage 5: Uncertainty ───────────────────────────────────────────────────────
def stage_uncertainty(processed, dl_results):
    import torch
    from src.uncertainty import mc_dropout_predict, ConformalPredictor, compute_calibration_curve
    logger.info("=" * 60); logger.info("STAGE 5: Uncertainty Quantification"); logger.info("=" * 60)
    device = _torch_seed(); loaders = dl_results["loaders"]
    test_x_list, test_y_list = [], []
    for xb, yb in loaders["test"]: test_x_list.append(xb); test_y_list.append(yb)
    test_x = torch.cat(test_x_list); test_y = torch.cat(test_y_list).numpy()
    cal_data = {}
    for mname in ["CNN1D", "HybridSTA"]:
        m = dl_results["models"].get(mname)
        if m is None: continue
        mc = mc_dropout_predict(m, test_x, n_samples=50, device=device)
        cal = compute_calibration_curve(mc["mean"], mc["std"], test_y)
        cal_data[f"MC Dropout ({mname})"] = cal
        dl_results["predictions"][mname]["y_std"] = mc["std"]
        logger.info("  MC Dropout %s ECE=%.4f", mname, cal["ece"])
    best = min(dl_results["results"], key=lambda k: dl_results["results"][k]["rmse"])
    bp   = dl_results["predictions"][best]
    n_cal = len(test_y) // 2
    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(bp["y_pred"][:n_cal], test_y[:n_cal])
    cov = cp.evaluate_coverage(bp["y_pred"][n_cal:], test_y[n_cal:])
    logger.info("  Conformal coverage=%.3f (target 0.90)", cov["empirical_coverage"])
    conf_levels = np.array([0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.97,0.99])
    obs = []
    for a in 1 - conf_levels:
        cp2 = ConformalPredictor(alpha=a); cp2.calibrate(bp["y_pred"][:n_cal], test_y[:n_cal])
        obs.append(cp2.evaluate_coverage(bp["y_pred"][n_cal:], test_y[n_cal:])["empirical_coverage"])
    cal_data["Conformal"] = {"expected_coverage": conf_levels, "observed_coverage": np.array(obs),
                              "ece": float(np.mean(np.abs(np.array(obs) - conf_levels)))}
    return {"calibration_data": cal_data, "conformal": cp}

# ── Stage 6: XAI ──────────────────────────────────────────────────────────────
def stage_xai(processed, baseline_res, dl_res):
    from src.explainability import compute_shap_values, rank_features_by_shap, compute_gradcam_for_samples, extract_attention_weights
    logger.info("=" * 60); logger.info("STAGE 6: Explainability"); logger.info("=" * 60)
    device = _torch_seed(); xai = {}
    if "XGBoost" in baseline_res["models"]:
        sr = compute_shap_values(baseline_res["models"]["XGBoost"],
                                  baseline_res["X_train"], baseline_res["X_test"][:200], model_type="tree")
        xai["shap_values"] = sr["shap_values"]
        xai["shap_ranking"] = rank_features_by_shap(sr["shap_values"], processed["feature_names"])
        logger.info("  Top-5 SHAP: %s", [r["feature"] for r in xai["shap_ranking"][:5]])
    if "CNN1D" in dl_res["models"]:
        xai["gradcam"] = compute_gradcam_for_samples(dl_res["models"]["CNN1D"],
                                                      processed["dataset"]["signals"],
                                                      processed["dataset"]["damage_state"], device=device)
    if "HybridSTA" in dl_res["models"]:
        xai["attention_weights"] = extract_attention_weights(dl_res["models"]["HybridSTA"],
                                                              processed["dataset"]["signals"][:200], device=device)
    return xai

# ── Stage 7: Optimization ─────────────────────────────────────────────────────
def stage_opt(dl_res):
    from src.optimization import MultiObjectiveOptimizer, create_objective_functions, decode_layup_vector
    logger.info("=" * 60); logger.info("STAGE 7: Bayesian Optimization"); logger.info("=" * 60)
    device = _torch_seed()
    obj_fns = create_objective_functions(dl_res["models"].get("CNN1D"), None, None, device=device)
    bounds  = np.array([[0.,1.]]*6)
    opt = MultiObjectiveOptimizer(obj_fns, bounds, n_initial=15, seed=SEED)
    res = opt.optimize(n_iterations=30)
    for cfg in opt.get_top_k_configs(k=3):
        logger.info("  Rank%d: %s RUL=%.3f", cfg["rank"], cfg["notation"], cfg["objectives"]["rul"])
    return {"pareto_front": res["pareto_front"], "all_points": res["Y_evaluated"],
            "top_configs": opt.get_top_k_configs(k=3), "optimizer": opt}

# ── Stage 8: Figures ───────────────────────────────────────────────────────────
def stage_viz(all_res):
    from src.visualization import (plot_raw_signals, plot_dwt_scalogram, plot_damage_index,
        plot_umap_projections, plot_correlation_heatmap, plot_radar_chart,
        plot_rul_predictions, plot_pinn_loss, plot_pinn_degradation,
        plot_attention_heatmap, plot_shap_beeswarm, plot_calibration_curves,
        plot_pareto_front, plot_confusion_matrices, plot_cv_boxplots)
    logger.info("=" * 60); logger.info("STAGE 8: Figures"); logger.info("=" * 60)
    out = "results/figures"; paths = []; ds = all_res["processed"]["dataset"]
    fns = [
        lambda: plot_raw_signals(ds["signals"], ds["damage_state"], output_dir=out),
        lambda: plot_dwt_scalogram(ds["signals"], ds["damage_state"], ds["cycles"], output_dir=out),
        lambda: plot_damage_index(all_res["processed"]["damage_index"], ds["cycles"], ds["layup_id"], output_dir=out),
        lambda: plot_umap_projections(all_res["processed"]["umap_embeddings"], ds["damage_state"], ds["layup_id"], ds["rul"], output_dir=out),
        lambda: plot_correlation_heatmap(all_res["processed"]["features"], all_res["processed"]["feature_names"], output_dir=out),
    ]
    for fn in fns:
        try: paths.append(fn())
        except Exception as e: logger.warning("Fig error: %s", e)
    # Radar
    try:
        all_metrics = {}
        for sk in ["baselines","dl"]:
            if sk in all_res:
                for mn, mv in all_res[sk]["results"].items():
                    all_metrics[mn] = {"RMSE↓": max(0,1-mv.get("rmse",0.5)),
                                        "MAE↓": max(0,1-mv.get("mae",0.5)),
                                        "R²↑": max(0,mv.get("r2",0))}
        if all_metrics: paths.append(plot_radar_chart(all_metrics, output_dir=out))
    except Exception as e: logger.warning("Radar: %s", e)
    # RUL preds
    try:
        rp = {}
        for sk in ["baselines","dl"]:
            if sk in all_res and "predictions" in all_res[sk]: rp.update(all_res[sk]["predictions"])
        if rp: paths.append(plot_rul_predictions(rp, output_dir=out))
    except Exception as e: logger.warning("RUL plot: %s", e)
    # PINN
    if "pinn" in all_res:
        try: paths.append(plot_pinn_loss(all_res["pinn"]["history"], output_dir=out))
        except Exception as e: logger.warning("PINN loss: %s", e)
        try:
            pr = all_res["pinn"]
            paths.append(plot_pinn_degradation(pr["cycles_norm"], pr["stiffness_true"],
                         pr["stiffness_pred"], pr["strength_true"], pr["strength_pred"], output_dir=out))
        except Exception as e: logger.warning("PINN deg: %s", e)
    if "xai" in all_res:
        try:
            if "attention_weights" in all_res["xai"] and all_res["xai"]["attention_weights"] is not None:
                paths.append(plot_attention_heatmap(all_res["xai"]["attention_weights"], ds["damage_state"][:200], output_dir=out))
        except Exception as e: logger.warning("Attn: %s", e)
        try:
            if "shap_values" in all_res["xai"]:
                sv = all_res["xai"]["shap_values"]
                paths.append(plot_shap_beeswarm(sv, all_res["processed"]["feature_names"],
                             all_res["processed"]["features"][:sv.shape[0]], output_dir=out))
        except Exception as e: logger.warning("SHAP: %s", e)
    if "uncertainty" in all_res:
        try: paths.append(plot_calibration_curves(all_res["uncertainty"]["calibration_data"], output_dir=out))
        except Exception as e: logger.warning("Cal: %s", e)
    if "optimization" in all_res and all_res["optimization"]["pareto_front"] is not None:
        try:
            pf = all_res["optimization"]["pareto_front"]
            Y  = all_res["optimization"]["all_points"]
            Ym = np.column_stack([np.array(v) for v in Y.values()])
            paths.append(plot_pareto_front(pf, Ym, output_dir=out))
        except Exception as e: logger.warning("Pareto: %s", e)
    logger.info("Generated %d figures.", len(paths))
    return paths

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CFRP AI Pipeline (Improved)")
    parser.add_argument("--data_path", default="./data")
    parser.add_argument("--mode", default="full_pipeline",
                        choices=["full_pipeline","baselines","deep_learning","pinn","visualization"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--force_raw", action="store_true")
    args = parser.parse_args()
    _dirs()
    logger.info("Mode: %s | Data: %s | Epochs: %d | Force Raw: %s", args.mode, args.data_path, args.epochs, args.force_raw)
    t0 = time.time()
    processed = stage_load_data(args.data_path, force_raw=args.force_raw)
    all_res = {"processed": processed}
    if args.mode in ["baselines","deep_learning","pinn","visualization","full_pipeline"]:
        all_res["baselines"] = stage_baselines(processed)
    if args.mode in ["deep_learning","pinn","visualization","full_pipeline"]:
        all_res["dl"] = stage_deep_learning(processed, epochs=args.epochs)
    if args.mode in ["pinn","visualization","full_pipeline"]:
        all_res["pinn"] = stage_pinn(processed)
    if args.mode in ["visualization","full_pipeline"]:
        all_res["uncertainty"] = stage_uncertainty(processed, all_res["dl"])
        all_res["xai"]         = stage_xai(processed, all_res["baselines"], all_res["dl"])
        all_res["optimization"] = stage_opt(all_res["dl"])
        stage_viz(all_res)
    logger.info("Pipeline complete in %.1f min", (time.time()-t0)/60)

if __name__ == "__main__":
    main()
