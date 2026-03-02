# src/pipeline.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, List, Tuple

import numpy as np

from config import ScenarioConfig
from src.baselines.weighted_kmeans import run_weighted_kmeans_baseline
from src.baselines.fast_beam_placement import run_bkmeans_baseline, run_tgbp_baseline
from src.evaluator import evaluate_cluster
from src.helper import summarize_multisat, print_summary, print_config
from src.plot import plot_clusters_overlay
from src.refine_qos_angle import refine_enterprise_by_angle
from src.refine_load_balance import refine_load_balance_by_overlap
from src.split import split_farthest
from src.profiling import Profiler
from src.usergen import generate_users, pack_users_raw, build_users_for_sat

from src.coords import unit
from src.satellites import sort_active_sats, associate_users_balanced


def _parse_time_utc_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _empty_summary() -> dict[str, Any]:
    return {
        "K": 0,
        "feasible_rate": 0.0,
        "U_mean": 0.0,
        "U_max": 0.0,
        "U_min": 0.0,
        "risk_sum": 0.0,
        "ent_total": 0,
        "ent_exposed": 0,
        "ent_edge_pct": 0.0,
        "ent_z_mean": 0.0,
        "ent_z_p90": 0.0,
        "ent_z_max": 0.0,
    }


def split_to_feasible(users, cfg: ScenarioConfig, prof: Profiler | None = None, max_clusters: int = 5000):
    pending = deque([np.arange(users.n, dtype=int)])
    clusters_feas: list[np.ndarray] = []
    evals_feas: list[dict] = []
    n_splits = 0

    while pending:
        if (len(pending) + len(clusters_feas)) > max_clusters:
            raise RuntimeError("Too many clusters. Check demands/PHY parameters.")

        S = pending.pop()
        if S.size == 0:
            continue

        ev = evaluate_cluster(users, S, cfg)
        if prof:
            prof.inc("eval_calls")

        if ev["feasible"]:
            clusters_feas.append(S)
            evals_feas.append(ev)
            continue

        seed = int(cfg.run.seed) + n_splits + 1
        S1, S2 = split_farthest(users.xy_m, S, seed=seed)
        S1 = np.asarray(S1, dtype=int)
        S2 = np.asarray(S2, dtype=int)

        if S1.size == 0 or S2.size == 0:
            raise RuntimeError(f"split_farthest produced empty split: |S|={S.size} -> {S1.size},{S2.size}")

        pending.append(S1)
        pending.append(S2)
        n_splits += 1

    if prof:
        prof.c["n_splits"] = prof.c.get("n_splits", 0) + n_splits

    return clusters_feas, evals_feas


def _elev_deg_users_to_sat(user_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> np.ndarray:
    up = unit(user_ecef_m)
    los = sat_ecef_m[None, :] - user_ecef_m
    los_hat = unit(los)
    sin_el = np.einsum("ij,ij->i", los_hat, up)
    sin_el = np.clip(sin_el, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_el)).astype(float)


def _payload_T_from_evals(evals: list[dict]) -> float:
    if not evals:
        return 0.0
    return float(np.sum([float(ev.get("U", 0.0)) for ev in evals]))


def _build_sat_piece(
    users_raw,
    sat_user_ids: List[np.ndarray],
    sat_ecef_m: np.ndarray,
    s_idx: int,
    cfg: ScenarioConfig,
    prof: Profiler,
) -> tuple[Any | None, list[np.ndarray] | None, list[dict] | None, float]:
    user_ids = sat_user_ids[s_idx]
    if user_ids.size == 0:
        return None, None, None, 0.0

    users_sat = build_users_for_sat(users_raw, user_ids, sat_ecef_m[s_idx])
    if users_sat.n == 0:
        return None, None, None, 0.0

    prof.tic("split")
    clusters, evals = split_to_feasible(users_sat, cfg, prof=prof)
    prof.toc("split")

    T_s = _payload_T_from_evals(evals)
    return users_sat, clusters, evals, float(T_s)


def _payload_repair_inner(
    users_raw,
    sat_ecef_m: np.ndarray,
    sat_user_ids_init: List[np.ndarray],
    cfg: ScenarioConfig,
    prof: Profiler,
) -> tuple[bool, List[np.ndarray], dict[int, tuple[Any, list[np.ndarray], list[dict]]], dict]:
    """
    Try to satisfy per-satellite payload time budget:
        T_s = sum_b U_{s,b} <= J_lanes

    Repair operator:
        Move the single highest-U beam (cluster) from an overloaded sat to a receiver sat
        where ALL users are visible and receiver has slack.
    """
    ms = cfg.multisat
    pcfg = cfg.payload

    J = float(pcfg.J_lanes)
    emin = float(ms.elev_mask_deg)
    tol = 1e-9

    S = int(sat_ecef_m.shape[0])
    sat_user_ids: List[np.ndarray] = [np.asarray(x, dtype=int).copy() for x in sat_user_ids_init]

    users_sat_cache: List[Any | None] = [None] * S
    clusters_cache: List[list[np.ndarray] | None] = [None] * S
    evals_cache: List[list[dict] | None] = [None] * S
    T_cache = np.zeros(S, dtype=float)

    main_by_sat: dict[int, tuple[Any, list[np.ndarray], list[dict]]] = {}

    # Build initial per-sat clustering
    for s in range(S):
        u, c, e, T = _build_sat_piece(users_raw, sat_user_ids, sat_ecef_m, s, cfg, prof)
        users_sat_cache[s], clusters_cache[s], evals_cache[s], T_cache[s] = u, c, e, T
        if u is not None and c is not None and e is not None and sat_user_ids[s].size > 0:
            main_by_sat[s] = (u, c, e)

    T_sum = float(T_cache.sum())
    global_cap = float(S) * J
    global_impossible = (T_sum > global_cap + 1e-6)

    stats = {
        "enabled": True,
        "J_lanes": J,
        "rounds": 0,
        "moves": 0,
        "feasible": False,
        "n_viol": int(np.sum(T_cache > (J + tol))),
        "T_max": float(T_cache.max()) if S > 0 else 0.0,
        "T_sum": float(T_sum),
        "global_cap": float(global_cap),
        "global_impossible": bool(global_impossible),
    }

    # If globally impossible, repairs cannot help. Return immediately.
    if global_impossible:
        return False, sat_user_ids, main_by_sat, stats

    # Repair rounds
    for rnd in range(int(pcfg.max_rounds)):
        stats["rounds"] = rnd + 1

        viol = np.where(T_cache > (J + tol))[0]
        if viol.size == 0:
            stats["feasible"] = True
            stats["n_viol"] = 0
            break

        # Process violators in descending overload
        overload = T_cache[viol] - J
        viol = viol[np.argsort(-overload)]

        moved_this_round = 0

        for s_donor in viol.tolist():
            if moved_this_round >= int(pcfg.max_offloads_per_round):
                break

            clusters_d = clusters_cache[s_donor]
            evals_d = evals_cache[s_donor]
            if clusters_d is None or evals_d is None or len(clusters_d) == 0:
                continue

            U_beams = np.array([float(ev.get("U", 0.0)) for ev in evals_d], dtype=float)
            k_beam = int(np.argmax(U_beams))
            if not np.isfinite(U_beams[k_beam]) or U_beams[k_beam] <= 0.0:
                continue

            donor_local = np.asarray(clusters_d[k_beam], dtype=int)
            donor_global = sat_user_ids[s_donor][donor_local]
            if donor_global.size == 0:
                continue

            best_t = -1
            best_score = -1e99

            for t in range(S):
                if t == s_donor:
                    continue

                # receiver slack
                slack = J - float(T_cache[t])
                if slack <= 0.0:
                    continue

                elev = _elev_deg_users_to_sat(users_raw.ecef_m[donor_global], sat_ecef_m[t])
                if np.any(elev < emin):
                    continue

                # prioritize slack; tie-break by mean elevation
                score = float(slack) + 0.01 * float(np.mean(elev))
                if score > best_score:
                    best_score = score
                    best_t = t

            if best_t < 0:
                continue

            donor_set = sat_user_ids[s_donor]
            moved_flags = np.isin(donor_set, donor_global, assume_unique=False)
            new_donor = donor_set[~moved_flags]
            new_recv = np.concatenate([sat_user_ids[best_t], donor_global], axis=0).astype(int, copy=False)

            sat_user_ids[s_donor] = new_donor
            sat_user_ids[best_t] = new_recv

            # rebuild donor + receiver only
            for s in (s_donor, best_t):
                u, c, e, T = _build_sat_piece(users_raw, sat_user_ids, sat_ecef_m, s, cfg, prof)
                users_sat_cache[s], clusters_cache[s], evals_cache[s], T_cache[s] = u, c, e, T
                if u is not None and c is not None and e is not None and sat_user_ids[s].size > 0:
                    main_by_sat[s] = (u, c, e)
                else:
                    main_by_sat.pop(s, None)

            stats["moves"] += 1
            moved_this_round += 1

        if moved_this_round == 0:
            break

    stats["n_viol"] = int(np.sum(T_cache > (J + tol)))
    stats["T_max"] = float(T_cache.max()) if S > 0 else 0.0
    stats["T_sum"] = float(T_cache.sum())
    stats["feasible"] = (stats["n_viol"] == 0)

    return bool(stats["feasible"]), sat_user_ids, main_by_sat, stats


@dataclass(frozen=True)
class _Candidate:
    m: int
    assoc: Any
    sat_user_ids: List[np.ndarray]
    main_by_sat: dict[int, tuple[Any, list[np.ndarray], list[dict]]]
    payload_stats: dict
    feasible: bool


def run_scenario(cfg: ScenarioConfig) -> dict[str, Any]:
    prof = Profiler()
    ms = cfg.multisat
    pcfg = cfg.payload

    # 1) Generate users
    prof.tic("usergen")
    user_list = generate_users(cfg)
    users_raw = pack_users_raw(user_list)
    prof.toc("usergen")

    verbose = bool(getattr(cfg.run, "verbose", True))
    enable_plots = bool(getattr(cfg.run, "enable_plots", False))
    enable_fastbp = bool(getattr(cfg.run, "enable_fastbp_baselines", True))

    print_config(cfg)

    # 2) Select satellites
    t0_utc = _parse_time_utc_iso(ms.time_utc_iso) if ms.time_utc_iso else None

    prof.tic("sat_select")
    t0_utc, active_sats = sort_active_sats(cfg, t0_utc=t0_utc, n_lat_anchors=3, n_lon_anchors=3, quality_mode="sin")
    prof.toc("sat_select")

    if len(active_sats) == 0:
        raise RuntimeError("No active satellites found above elev mask at any anchor in the region bbox.")

    sat_ecef_full = np.stack([s.ecef_m for s in active_sats], axis=0)

    # 3) Prefix scan
    max_prefix = int(pcfg.max_prefix) if pcfg.max_prefix is not None else int(len(active_sats))
    max_prefix = max(1, min(max_prefix, int(len(active_sats))))

    best_feasible: Optional[_Candidate] = None
    best_failed: Optional[_Candidate] = None

    for m in range(1, max_prefix + 1):
        sat_ecef_m = sat_ecef_full[:m].copy()

        # association for this prefix
        prof.tic("assoc")
        assoc = associate_users_balanced(
            user_ecef_m=users_raw.ecef_m,
            user_demand_mbps=users_raw.demand_mbps,
            user_qos_w=users_raw.qos_w,
            sat_ecef_m=sat_ecef_m,
            elev_mask_deg=ms.elev_mask_deg,
            load_mode=ms.assoc_load_mode,
            slack=ms.assoc_slack,
            max_rounds=ms.assoc_max_rounds,
            max_total_moves=ms.assoc_max_moves,
            seed=int(cfg.run.seed),
        )
        prof.toc("assoc")

        sat_user_ids_init = [np.asarray(x, dtype=int) for x in assoc.sat_user_ids]

        if not pcfg.enabled:
            # build main_by_sat once
            main_by_sat: dict[int, tuple[Any, list[np.ndarray], list[dict]]] = {}
            for s in range(m):
                u, c, e, _T = _build_sat_piece(users_raw, sat_user_ids_init, sat_ecef_m, s, cfg, prof)
                if u is not None and c is not None and e is not None and sat_user_ids_init[s].size > 0:
                    main_by_sat[s] = (u, c, e)
            feasible = True
            payload_stats = {"enabled": False, "feasible": True, "n_viol": 0, "T_max": 0.0, "T_sum": 0.0}
            sat_user_ids_out = sat_user_ids_init
        else:
            feasible, sat_user_ids_out, main_by_sat, payload_stats = _payload_repair_inner(
                users_raw=users_raw,
                sat_ecef_m=sat_ecef_m,
                sat_user_ids_init=sat_user_ids_init,
                cfg=cfg,
                prof=prof,
            )

        cand = _Candidate(
            m=m,
            assoc=assoc,
            sat_user_ids=sat_user_ids_out,
            main_by_sat=main_by_sat,
            payload_stats=payload_stats,
            feasible=bool(feasible),
        )

        if cand.feasible:
            best_feasible = cand
            break

        # keep best failed (fewest viol, then smallest Tmax, then smaller m)
        if best_failed is None:
            best_failed = cand
        else:
            a = best_failed.payload_stats
            b = cand.payload_stats
            key_a = (int(a.get("n_viol", 10**9)), float(a.get("T_max", 1e99)), int(best_failed.m))
            key_b = (int(b.get("n_viol", 10**9)), float(b.get("T_max", 1e99)), int(cand.m))
            if key_b < key_a:
                best_failed = cand

    chosen = best_feasible if best_feasible is not None else best_failed
    if chosen is None:
        # Should never happen, but keep sweep safe
        chosen = _Candidate(
            m=max_prefix,
            assoc=type("X", (), {"n_unserved": users_raw.n, "n_moves": 0})(),
            sat_user_ids=[np.array([], dtype=int) for _ in range(max_prefix)],
            main_by_sat={},
            payload_stats={"enabled": pcfg.enabled, "feasible": False, "n_viol": max_prefix, "T_max": np.nan, "T_sum": np.nan},
            feasible=False,
        )

    payload_feasible = bool(chosen.feasible)
    m_used = int(chosen.m)

    # Build pieces_main from chosen main_by_sat
    pieces_main: list[tuple[Any, list[np.ndarray], list[dict]]] = [chosen.main_by_sat[s] for s in sorted(chosen.main_by_sat.keys())]

    # If payload infeasible -> return early (skip refinements/baselines to keep Phase B alive)
    if not payload_feasible:
        main_summary = summarize_multisat(pieces_main, cfg) if pieces_main else _empty_summary()

        return {
            "seed": cfg.run.seed,
            "region_mode": cfg.region_mode,
            "n_users": cfg.run.n_users,

            "use_hotspots": cfg.usergen.enabled,
            "n_hotspots": cfg.usergen.n_hotspots,
            "noise_frac": cfg.usergen.noise_frac,
            "sigma_min": cfg.usergen.hotspot_sigma_m_min,
            "sigma_max": cfg.usergen.hotspot_sigma_m_max,

            "rho_safe": cfg.ent.rho_safe,
            "eirp_dbw": cfg.phy.eirp_dbw,
            "bandwidth_hz": cfg.phy.bandwidth_hz,
            "radius_modes_km": cfg.beam.radius_modes_km,

            # Multi-sat metadata
            "ms_tle_path": ms.tle_path,
            "ms_time_utc": t0_utc.isoformat(),
            "ms_elev_mask_deg": ms.elev_mask_deg,
            "ms_n_active": int(m_used),
            "ms_n_unserved": int(getattr(chosen.assoc, "n_unserved", 0)),
            "ms_assoc_moves": int(getattr(chosen.assoc, "n_moves", 0)),

            # Payload metadata
            "payload_enabled": bool(pcfg.enabled),
            "payload_J_lanes": float(pcfg.J_lanes),
            "payload_feasible": False,
            "payload_best_m": int(m_used),
            "payload_n_viol": int(chosen.payload_stats.get("n_viol", 0)),
            "payload_T_max": float(chosen.payload_stats.get("T_max", 0.0)),
            "payload_T_sum": float(chosen.payload_stats.get("T_sum", 0.0)),
            "payload_global_cap": float(chosen.payload_stats.get("global_cap", float(m_used) * float(pcfg.J_lanes))),
            "payload_global_impossible": bool(chosen.payload_stats.get("global_impossible", False)),

            # Summaries
            "main": main_summary,
            "main_ref": _empty_summary(),
            "main_ref_lb": _empty_summary(),
            "wk_demand_fixed": _empty_summary(),
            "wk_demand_rep": _empty_summary(),
            "wk_qos_fixed": _empty_summary(),
            "wk_qos_rep": _empty_summary(),
            "bk_fixed": _empty_summary(),
            "bk_rep": _empty_summary(),
            "tgbp_fixed": _empty_summary(),
            "tgbp_rep": _empty_summary(),

            # Timings
            "time_usergen_s": prof.t.get("usergen", 0.0),
            "time_sat_select_s": prof.t.get("sat_select", 0.0),
            "time_assoc_s": prof.t.get("assoc", 0.0),
            "time_split_s": prof.t.get("split", 0.0),
            "time_ent_ref_s": 0.0,
            "time_lb_ref_s": 0.0,

            "eval_calls": prof.c.get("eval_calls", 0),
            "n_splits": prof.c.get("n_splits", 0),
            "ent_moves_tried": 0,
            "ent_moves_accepted": 0,
            "lb_moves_tried": 0,
            "lb_moves_accepted": 0,

            "time_baseline_without_qos_s": 0.0,
            "time_baseline_with_qos_s": 0.0,
            "time_baseline_bkmeans_s": 0.0,
            "time_baseline_tgbp_s": 0.0,
        }

    # -----------------------------
    # Payload-feasible path: refinements + baselines
    # -----------------------------
    pieces_ref: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_lb: list[tuple[Any, list[np.ndarray], list[dict]]] = []

    pieces_wk_demand_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_wk_demand_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_wk_qos_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_wk_qos_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []

    pieces_bk_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_bk_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_tgbp_fixed: list[tuple[Any, list[np.ndarray], list[dict]]] = []
    pieces_tgbp_rep: list[tuple[Any, list[np.ndarray], list[dict]]] = []

    busiest_sat_idx = 0
    busiest_users = None
    busiest_clusters = busiest_evals = None
    busiest_clusters_ref = busiest_evals_ref = None
    busiest_clusters_lb = busiest_evals_lb = None

    # determine busiest sat (by users)
    sizes = [len(x) for x in chosen.sat_user_ids]
    if sizes:
        busiest_sat_idx = int(np.argmax(sizes))

    # process each sat in chosen set
    sat_ecef_m = sat_ecef_full[:m_used].copy()
    for s_idx in range(m_used):
        user_ids = chosen.sat_user_ids[s_idx]
        if user_ids.size == 0:
            continue

        users_sat = build_users_for_sat(users_raw, user_ids, sat_ecef_m[s_idx])
        if users_sat.n == 0:
            continue

        # main (rebuild once to keep consistent)
        prof.tic("split")
        clusters, evals = split_to_feasible(users_sat, cfg, prof=prof)
        prof.toc("split")
        # main pieces already exist; but keep consistent for refinements/baselines
        # (optional: could reuse chosen.main_by_sat)
        # pieces_main will be recomputed for summary below, so keep local only.

        # QoS refine
        prof.tic("ent_ref")
        clusters_ref, evals_ref, ref_stats = refine_enterprise_by_angle(
            users_sat, cfg, clusters, evals,
            n_rounds=cfg.qos_refine.rounds,
            kcand=cfg.qos_refine.kcand,
            max_moves_per_round=cfg.qos_refine.max_moves_per_round,
        )
        prof.toc("ent_ref")
        prof.c["ent_moves_tried"] = prof.c.get("ent_moves_tried", 0) + int(ref_stats.get("moves_tried", 0))
        prof.c["ent_moves_accepted"] = prof.c.get("ent_moves_accepted", 0) + int(ref_stats.get("moves_accepted", 0))
        pieces_ref.append((users_sat, clusters_ref, evals_ref))

        # LB refine
        prof.tic("lb_ref")
        clusters_lb, evals_lb, lb_stats = refine_load_balance_by_overlap(users_sat, cfg, clusters_ref, evals_ref)
        prof.toc("lb_ref")
        prof.c["lb_moves_tried"] = prof.c.get("lb_moves_tried", 0) + int(lb_stats.get("moves_tried", 0))
        prof.c["lb_moves_accepted"] = prof.c.get("lb_moves_accepted", 0) + int(lb_stats.get("moves_accepted", 0))
        pieces_lb.append((users_sat, clusters_lb, evals_lb))

        # Baselines use fixed K_ref = |clusters|
        K_ref = len(clusters)

        prof.tic("baseline_without_qos")
        base_wo = run_weighted_kmeans_baseline(users_sat, cfg, K_ref=K_ref, use_qos_weight=False)
        prof.toc("baseline_without_qos")
        pieces_wk_demand_fixed.append((users_sat, base_wo["fixedK"]["clusters"], base_wo["fixedK"]["evals"]))
        pieces_wk_demand_rep.append((users_sat, base_wo["repaired"]["clusters"], base_wo["repaired"]["evals"]))

        prof.tic("baseline_with_qos")
        base_wq = run_weighted_kmeans_baseline(users_sat, cfg, K_ref=K_ref, use_qos_weight=True)
        prof.toc("baseline_with_qos")
        pieces_wk_qos_fixed.append((users_sat, base_wq["fixedK"]["clusters"], base_wq["fixedK"]["evals"]))
        pieces_wk_qos_rep.append((users_sat, base_wq["repaired"]["clusters"], base_wq["repaired"]["evals"]))

        if enable_fastbp:
            prof.tic("baseline_bkmeans")
            out_bk = run_bkmeans_baseline(users_sat, cfg, K_hint=K_ref)
            prof.toc("baseline_bkmeans")
            pieces_bk_fixed.append((users_sat, out_bk["fixedK"]["clusters"], out_bk["fixedK"]["evals"]))
            pieces_bk_rep.append((users_sat, out_bk["repaired"]["clusters"], out_bk["repaired"]["evals"]))

            prof.tic("baseline_tgbp")
            out_tg = run_tgbp_baseline(users_sat, cfg)
            prof.toc("baseline_tgbp")
            pieces_tgbp_fixed.append((users_sat, out_tg["fixedK"]["clusters"], out_tg["fixedK"]["evals"]))
            pieces_tgbp_rep.append((users_sat, out_tg["repaired"]["clusters"], out_tg["repaired"]["evals"]))

        if s_idx == busiest_sat_idx:
            busiest_users = users_sat
            busiest_clusters, busiest_evals = clusters, evals
            busiest_clusters_ref, busiest_evals_ref = clusters_ref, evals_ref
            busiest_clusters_lb, busiest_evals_lb = clusters_lb, evals_lb

    # main summary from chosen solution
    main_summary = summarize_multisat(pieces_main, cfg) if pieces_main else _empty_summary()
    ref_summary = summarize_multisat(pieces_ref, cfg) if pieces_ref else _empty_summary()
    lb_summary = summarize_multisat(pieces_lb, cfg) if pieces_lb else _empty_summary()

    wk_demand_fixed = summarize_multisat(pieces_wk_demand_fixed, cfg) if pieces_wk_demand_fixed else _empty_summary()
    wk_demand_rep = summarize_multisat(pieces_wk_demand_rep, cfg) if pieces_wk_demand_rep else _empty_summary()
    wk_qos_fixed = summarize_multisat(pieces_wk_qos_fixed, cfg) if pieces_wk_qos_fixed else _empty_summary()
    wk_qos_rep = summarize_multisat(pieces_wk_qos_rep, cfg) if pieces_wk_qos_rep else _empty_summary()

    bk_fixed = summarize_multisat(pieces_bk_fixed, cfg) if pieces_bk_fixed else _empty_summary()
    bk_rep = summarize_multisat(pieces_bk_rep, cfg) if pieces_bk_rep else _empty_summary()
    tgbp_fixed = summarize_multisat(pieces_tgbp_fixed, cfg) if pieces_tgbp_fixed else _empty_summary()
    tgbp_rep = summarize_multisat(pieces_tgbp_rep, cfg) if pieces_tgbp_rep else _empty_summary()

    if verbose:
        print_summary("Main algorithm (payload-feasible) [global]", main_summary, cfg)
        print_summary("Main + enterprise refinement [global]", ref_summary, cfg)
        print_summary("Main + enterprise + load-balance [global]", lb_summary, cfg)

    if enable_plots and busiest_users is not None and busiest_clusters is not None:
        plot_clusters_overlay(
            users_xy_m=busiest_users.xy_m,
            qos_w=busiest_users.qos_w,
            clusters=busiest_clusters,
            evals=busiest_evals,
            title=f"Busiest sat: main (K={len(busiest_clusters)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
        plot_clusters_overlay(
            users_xy_m=busiest_users.xy_m,
            qos_w=busiest_users.qos_w,
            clusters=busiest_clusters_ref,
            evals=busiest_evals_ref,
            title=f"Busiest sat: main+QoS refine (K={len(busiest_clusters_ref)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )
        plot_clusters_overlay(
            users_xy_m=busiest_users.xy_m,
            qos_w=busiest_users.qos_w,
            clusters=busiest_clusters_lb,
            evals=busiest_evals_lb,
            title=f"Busiest sat: main+QoS+LB refine (K={len(busiest_clusters_lb)})",
            draw_circles=True,
            draw_centers=True,
            max_circles=250,
        )

    return {
        "seed": cfg.run.seed,
        "region_mode": cfg.region_mode,
        "n_users": cfg.run.n_users,

        "use_hotspots": cfg.usergen.enabled,
        "n_hotspots": cfg.usergen.n_hotspots,
        "noise_frac": cfg.usergen.noise_frac,
        "sigma_min": cfg.usergen.hotspot_sigma_m_min,
        "sigma_max": cfg.usergen.hotspot_sigma_m_max,

        "rho_safe": cfg.ent.rho_safe,
        "eirp_dbw": cfg.phy.eirp_dbw,
        "bandwidth_hz": cfg.phy.bandwidth_hz,
        "radius_modes_km": cfg.beam.radius_modes_km,

        "ms_tle_path": ms.tle_path,
        "ms_time_utc": t0_utc.isoformat(),
        "ms_elev_mask_deg": ms.elev_mask_deg,
        "ms_n_active": int(m_used),
        "ms_n_unserved": int(getattr(chosen.assoc, "n_unserved", 0)),
        "ms_assoc_moves": int(getattr(chosen.assoc, "n_moves", 0)),

        "payload_enabled": bool(pcfg.enabled),
        "payload_J_lanes": float(pcfg.J_lanes),
        "payload_feasible": True,
        "payload_best_m": int(m_used),
        "payload_n_viol": int(chosen.payload_stats.get("n_viol", 0)),
        "payload_T_max": float(chosen.payload_stats.get("T_max", 0.0)),
        "payload_T_sum": float(chosen.payload_stats.get("T_sum", 0.0)),
        "payload_global_cap": float(chosen.payload_stats.get("global_cap", float(m_used) * float(pcfg.J_lanes))),
        "payload_global_impossible": bool(chosen.payload_stats.get("global_impossible", False)),

        "main": main_summary,
        "main_ref": ref_summary,
        "main_ref_lb": lb_summary,

        "wk_demand_fixed": wk_demand_fixed,
        "wk_demand_rep": wk_demand_rep,
        "wk_qos_fixed": wk_qos_fixed,
        "wk_qos_rep": wk_qos_rep,

        "bk_fixed": bk_fixed,
        "bk_rep": bk_rep,
        "tgbp_fixed": tgbp_fixed,
        "tgbp_rep": tgbp_rep,

        "time_usergen_s": prof.t.get("usergen", 0.0),
        "time_sat_select_s": prof.t.get("sat_select", 0.0),
        "time_assoc_s": prof.t.get("assoc", 0.0),
        "time_split_s": prof.t.get("split", 0.0),
        "time_ent_ref_s": prof.t.get("ent_ref", 0.0),
        "time_lb_ref_s": prof.t.get("lb_ref", 0.0),

        "eval_calls": prof.c.get("eval_calls", 0),
        "n_splits": prof.c.get("n_splits", 0),
        "ent_moves_tried": prof.c.get("ent_moves_tried", 0),
        "ent_moves_accepted": prof.c.get("ent_moves_accepted", 0),
        "lb_moves_tried": prof.c.get("lb_moves_tried", 0),
        "lb_moves_accepted": prof.c.get("lb_moves_accepted", 0),

        "time_baseline_without_qos_s": prof.t.get("baseline_without_qos", 0.0),
        "time_baseline_with_qos_s": prof.t.get("baseline_with_qos", 0.0),
        "time_baseline_bkmeans_s": prof.t.get("baseline_bkmeans", 0.0),
        "time_baseline_tgbp_s": prof.t.get("baseline_tgbp", 0.0),
    }