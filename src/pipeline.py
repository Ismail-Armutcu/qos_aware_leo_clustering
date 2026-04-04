# src/pipeline.py
from __future__ import annotations


from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, List


import numpy as np


from config import ScenarioConfig
from src.baselines.weighted_kmeans import run_weighted_kmeans_baseline
from src.baselines.fast_beam_placement import run_bkmeans_baseline, run_tgbp_baseline
from src.evaluator import evaluate_cluster
from src.helper import summarize_multisat, print_summary, print_config
from src.plot import plot_clusters_overlay, plot_payload_sat_stats
from src.refine_qos_angle import refine_enterprise_by_angle
from src.refine_load_balance import refine_load_balance_by_overlap
from src.split import split_farthest
from src.profiling import Profiler
from src.usergen import generate_users, pack_users_raw, build_users_for_sat


from src.coords import unit
from src.satellites import sort_active_sats, associate_users_by_rule




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


       "radius_mean_km": 0.0,
       "radius_min_km": 0.0,
       "radius_p10_km": 0.0,
       "radius_p50_km": 0.0,
       "radius_p90_km": 0.0,
       "radius_max_km": 0.0,


       "sat_mean_radius_mean_km": 0.0,
       "sat_mean_radius_min_km": 0.0,
       "sat_mean_radius_max_km": 0.0,
   }




def _nan_summary() -> dict[str, Any]:
   nan = float("nan")
   return {
       "K": nan,
       "feasible_rate": nan,
       "U_mean": nan,
       "U_max": nan,
       "U_min": nan,
       "risk_sum": nan,
       "ent_total": nan,
       "ent_exposed": nan,
       "ent_edge_pct": nan,
       "ent_z_mean": nan,
       "ent_z_p90": nan,
       "ent_z_max": nan,


       "radius_mean_km": nan,
       "radius_min_km": nan,
       "radius_p10_km": nan,
       "radius_p50_km": nan,
       "radius_p90_km": nan,
       "radius_max_km": nan,


       "sat_mean_radius_mean_km": nan,
       "sat_mean_radius_min_km": nan,
       "sat_mean_radius_max_km": nan,
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
   vals = []
   for ev in evals:
       try:
           u = float(ev.get("U", 0.0))
       except Exception:
           return float("inf")
       if not np.isfinite(u) or u < 0.0:
           return float("inf")
       vals.append(u)
   return float(np.sum(vals))




def _build_sat_piece_with_local_builder(
   users_raw,
   sat_user_ids: List[np.ndarray],
   sat_ecef_m: np.ndarray,
   s_idx: int,
   cfg: ScenarioConfig,
   prof: Profiler,
   local_builder,
) -> tuple[Any | None, list[np.ndarray] | None, list[dict] | None, float]:
   user_ids = sat_user_ids[s_idx]
   if user_ids.size == 0:
       return None, None, None, 0.0


   users_sat = build_users_for_sat(users_raw, user_ids, sat_ecef_m[s_idx])
   if users_sat.n == 0:
       return None, None, None, 0.0


   clusters, evals, _stats = local_builder(users_sat, cfg, prof)
   T_s = _payload_T_from_evals(evals)
   return users_sat, clusters, evals, float(T_s)




def _local_build_main(users_sat, cfg: ScenarioConfig, prof: Profiler):
   prof.tic("split")
   clusters, evals = split_to_feasible(users_sat, cfg, prof=prof)
   prof.toc("split")
   return clusters, evals, {}




def _local_build_main_ref(users_sat, cfg: ScenarioConfig, prof: Profiler):
   prof.tic("split")
   clusters, evals = split_to_feasible(users_sat, cfg, prof=prof)
   prof.toc("split")


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
   return clusters_ref, evals_ref, ref_stats




def _local_build_main_ref_lb(users_sat, cfg: ScenarioConfig, prof: Profiler):
   clusters_ref, evals_ref, _ = _local_build_main_ref(users_sat, cfg, prof)
   prof.tic("lb_ref")
   clusters_lb, evals_lb, lb_stats = refine_load_balance_by_overlap(users_sat, cfg, clusters_ref, evals_ref)
   prof.toc("lb_ref")
   prof.c["lb_moves_tried"] = prof.c.get("lb_moves_tried", 0) + int(lb_stats.get("moves_tried", 0))
   prof.c["lb_moves_accepted"] = prof.c.get("lb_moves_accepted", 0) + int(lb_stats.get("moves_accepted", 0))
   return clusters_lb, evals_lb, lb_stats




def _local_build_wkmeans_repaired(use_qos_weight: bool):
   cat = "baseline_with_qos" if use_qos_weight else "baseline_without_qos"
   def _builder(users_sat, cfg: ScenarioConfig, prof: Profiler):
       K_ref = len(split_to_feasible(users_sat, cfg, prof=None)[0])
       prof.tic(cat)
       out = run_weighted_kmeans_baseline(users_sat, cfg, K_ref=K_ref, use_qos_weight=use_qos_weight)
       prof.toc(cat)
       rep = out["repaired"]
       return rep["clusters"], rep["evals"], {}
   return _builder




def _local_build_bkmeans_repaired(users_sat, cfg: ScenarioConfig, prof: Profiler):
   K_ref = len(split_to_feasible(users_sat, cfg, prof=None)[0])
   prof.tic("baseline_bkmeans")
   out = run_bkmeans_baseline(users_sat, cfg, K_hint=K_ref)
   prof.toc("baseline_bkmeans")
   rep = out["repaired"]
   return rep["clusters"], rep["evals"], {}




def _local_build_tgbp_repaired(users_sat, cfg: ScenarioConfig, prof: Profiler):
   prof.tic("baseline_tgbp")
   out = run_tgbp_baseline(users_sat, cfg)
   prof.toc("baseline_tgbp")
   rep = out["repaired"]
   return rep["clusters"], rep["evals"], {}




def _payload_repair_inner(
   users_raw,
   sat_ecef_m: np.ndarray,
   sat_user_ids_init: List[np.ndarray],
   cfg: ScenarioConfig,
   prof: Profiler,
   local_builder,
) -> tuple[bool, List[np.ndarray], dict[int, tuple[Any, list[np.ndarray], list[dict]]], dict]:
   import math


   ms = cfg.multisat
   pcfg = cfg.payload


   J = float(pcfg.J_lanes)
   W = int(pcfg.W_slots)
   K_cap = int(pcfg.Ks_max)
   T_cap = float(J * float(W))


   emin = float(ms.elev_mask_deg)
   tol = 1e-9


   S = int(sat_ecef_m.shape[0])
   sat_user_ids: List[np.ndarray] = [np.asarray(x, dtype=int).copy() for x in sat_user_ids_init]


   clusters_cache: List[list[np.ndarray] | None] = [None] * S
   evals_cache: List[list[dict] | None] = [None] * S


   T_cache = np.zeros(S, dtype=float)
   K_cache = np.zeros(S, dtype=int)


   main_by_sat: dict[int, tuple[Any, list[np.ndarray], list[dict]]] = {}


   def w_min_req_from_Tmax(T_max: float) -> int:
       T_max = float(T_max)
       if not np.isfinite(T_max):
           return int(W + 1)
       return int(math.ceil(T_max / max(J, 1e-12)))


   def rebuild_sat(s: int) -> bool:
       try:
           u, c, e, T = _build_sat_piece_with_local_builder(users_raw, sat_user_ids, sat_ecef_m, s, cfg, prof, local_builder)
       except Exception:
           u, c, e, T = None, None, None, float("inf")


       T = float(T)
       valid = np.isfinite(T)
       if e is not None:
           try:
               valid = valid and all(np.isfinite(float(ev.get("U", 0.0))) and float(ev.get("U", 0.0)) >= 0.0 for ev in e)
           except Exception:
               valid = False


       if not valid:
           clusters_cache[s] = None
           evals_cache[s] = None
           T_cache[s] = float("inf")
           K_cache[s] = int(K_cap + 1)
           main_by_sat.pop(s, None)
           return False


       clusters_cache[s] = c
       evals_cache[s] = e
       T_cache[s] = T
       K_cache[s] = int(len(c)) if c is not None else 0
       if u is not None and c is not None and e is not None and sat_user_ids[s].size > 0:
           main_by_sat[s] = (u, c, e)
       else:
           main_by_sat.pop(s, None)
       return True


   for s in range(S):
       rebuild_sat(s)


   if np.any(~np.isfinite(T_cache)):
       stats = {
           "enabled": True,
           "J_lanes": J,
           "W_slots": W,
           "Ks_max": K_cap,
           "T_cap": T_cap,
           "K_cap": K_cap,
           "rounds": 0,
           "moves_tried": 0,
           "moves_accepted": 0,
           "moves_tried_K": 0,
           "moves_accepted_K": 0,
           "moves_tried_T": 0,
           "moves_accepted_T": 0,
           "moves_tried_smooth": 0,
           "moves_accepted_smooth": 0,
           "feasible": False,
           "n_viol_T": int(np.sum(np.isfinite(T_cache) & ((T_cache - T_cap) > tol))) + int(np.sum(~np.isfinite(T_cache))),
           "n_viol_K": int(np.sum((K_cache - K_cap) > tol)),
           "T_over_sum": float(np.sum(np.where(np.isfinite(T_cache), np.maximum(0.0, T_cache - T_cap), T_cap))),
           "K_over_sum": float(np.sum(np.maximum(0.0, K_cache.astype(float) - float(K_cap)))),
           "T_over_max": float(np.max(np.where(np.isfinite(T_cache), np.maximum(0.0, T_cache - T_cap), T_cap))) if S else 0.0,
           "K_over_max": float(np.max(np.maximum(0.0, K_cache.astype(float) - float(K_cap)))) if S else 0.0,
           "T_sum": float(np.sum(np.where(np.isfinite(T_cache), T_cache, T_cap))),
           "T_max": float(np.max(np.where(np.isfinite(T_cache), T_cache, T_cap))) if S else 0.0,
           "K_sum": int(np.sum(K_cache)),
           "K_max": int(np.max(K_cache)) if S else 0,
           "W_min_req": int(W + 1),
           "global_cap": float(float(S) * T_cap),
           "global_impossible": False,
           "build_invalid": True,
           "T_by_sat": T_cache.copy(),
           "K_by_sat": K_cache.copy(),
       }
       return False, sat_user_ids, main_by_sat, stats


   T_sum = float(np.sum(T_cache))
   global_cap = float(S) * T_cap
   if T_sum > global_cap + 1e-6:
       stats = {
           "enabled": True,
           "J_lanes": J,
           "W_slots": W,
           "Ks_max": K_cap,
           "T_cap": T_cap,
           "K_cap": K_cap,
           "rounds": 0,
           "moves_tried": 0,
           "moves_accepted": 0,
           "moves_tried_K": 0,
           "moves_accepted_K": 0,
           "moves_tried_T": 0,
           "moves_accepted_T": 0,
           "moves_tried_smooth": 0,
           "moves_accepted_smooth": 0,
           "feasible": False,
           "n_viol_T": int(np.sum((T_cache - T_cap) > tol)),
           "n_viol_K": int(np.sum((K_cache - K_cap) > tol)),
           "T_over_sum": float(np.sum(np.maximum(0.0, T_cache - T_cap))),
           "K_over_sum": float(np.sum(np.maximum(0.0, K_cache.astype(float) - float(K_cap)))),
           "T_over_max": float(np.max(np.maximum(0.0, T_cache - T_cap))) if S else 0.0,
           "K_over_max": float(np.max(np.maximum(0.0, K_cache.astype(float) - float(K_cap)))) if S else 0.0,
           "T_sum": float(T_sum),
           "T_max": float(np.max(T_cache)) if S else 0.0,
           "K_sum": int(np.sum(K_cache)),
           "K_max": int(np.max(K_cache)) if S else 0,
           "W_min_req": int(w_min_req_from_Tmax(float(np.max(T_cache)) if S else 0.0)),
           "global_cap": float(global_cap),
           "global_impossible": True,
           "T_by_sat": T_cache.copy(),
           "K_by_sat": K_cache.copy(),
       }
       return False, sat_user_ids, main_by_sat, stats


   def state_key() -> tuple:
       over_K = np.maximum(0.0, K_cache.astype(float) - float(K_cap))
       over_T = np.maximum(0.0, T_cache - T_cap)
       nK = int(np.sum(over_K > tol))
       nT = int(np.sum(over_T > tol))
       Ksum = float(np.sum(over_K))
       Tsum = float(np.sum(over_T))
       Kmax = float(np.max(over_K)) if S else 0.0
       Tmax = float(np.max(over_T)) if S else 0.0
       Wreq = int(w_min_req_from_Tmax(float(np.max(T_cache)) if S else 0.0))
       Wgap = int(max(0, Wreq - W))
       return (nK, Ksum, Kmax, nT, Wgap, Tsum, Tmax)


   def choose_receiver(donor_global: np.ndarray, s_donor: int, mode: str) -> int:
       best_t = -1
       best_score = -1e99
       for t in range(S):
           if t == s_donor:
               continue
           if K_cache[t] >= K_cap:
               continue
           slack_T = float(T_cap - T_cache[t])
           if slack_T <= 0.0:
               continue
           elev = _elev_deg_users_to_sat(users_raw.ecef_m[donor_global], sat_ecef_m[t])
           if np.any(elev < emin):
               continue
           slack_K = float(K_cap - int(K_cache[t]))
           if mode == "K":
               score = 100.0 * slack_K + 1.0 * slack_T + 0.05 * float(np.mean(elev))
           elif mode == "T":
               score = 100.0 * slack_T + 5.0 * slack_K + 0.05 * float(np.mean(elev))
           else:
               score = 100.0 * slack_T + 10.0 * slack_K + 0.05 * float(np.mean(elev))
           if score > best_score:
               best_score = score
               best_t = int(t)
       return int(best_t)


   moves_tried = moves_accepted = 0
   moves_tried_K = moves_accepted_K = 0
   moves_tried_T = moves_accepted_T = 0
   moves_tried_smooth = moves_accepted_smooth = 0


   def try_move(s_donor: int, beam_local_ids: np.ndarray, mode: str, accept_predicate=None) -> bool:
       nonlocal moves_tried, moves_accepted, moves_tried_K, moves_accepted_K, moves_tried_T, moves_accepted_T, moves_tried_smooth, moves_accepted_smooth


       if beam_local_ids.size == 0:
           return False


       nloc = int(sat_user_ids[s_donor].size)
       if np.any(beam_local_ids < 0) or np.any(beam_local_ids >= nloc):
           return False


       donor_global = sat_user_ids[s_donor][beam_local_ids]
       if donor_global.size == 0:
           return False


       s_recv = choose_receiver(donor_global, s_donor, mode=mode)
       if s_recv < 0:
           return False


       donor_old = sat_user_ids[s_donor].copy()
       recv_old = sat_user_ids[s_recv].copy()


       key_before = state_key()
       Tmax_before = float(np.max(T_cache)) if S else 0.0
       Wreq_before = w_min_req_from_Tmax(Tmax_before)


       moved_flags = np.isin(donor_old, donor_global, assume_unique=False)
       sat_user_ids[s_donor] = donor_old[~moved_flags]
       sat_user_ids[s_recv] = np.concatenate([recv_old, donor_global], axis=0).astype(int, copy=False)


       moves_tried += 1
       if mode == "K":
           moves_tried_K += 1
       elif mode == "T":
           moves_tried_T += 1
       else:
           moves_tried_smooth += 1


       ok1 = rebuild_sat(s_donor)
       ok2 = rebuild_sat(s_recv)
       if not (ok1 and ok2):
           sat_user_ids[s_donor] = donor_old
           sat_user_ids[s_recv] = recv_old
           rebuild_sat(s_donor)
           rebuild_sat(s_recv)
           return False


       key_after = state_key()
       Tmax_after = float(np.max(T_cache)) if S else 0.0
       Wreq_after = w_min_req_from_Tmax(Tmax_after)


       if accept_predicate is None:
           accept = (key_after < key_before)
       else:
           try:
               accept = bool(accept_predicate(key_before, key_after, Wreq_before, Wreq_after))
           except Exception:
               accept = False


       if accept:
           moves_accepted += 1
           if mode == "K":
               moves_accepted_K += 1
           elif mode == "T":
               moves_accepted_T += 1
           else:
               moves_accepted_smooth += 1
           return True


       sat_user_ids[s_donor] = donor_old
       sat_user_ids[s_recv] = recv_old
       rebuild_sat(s_donor)
       rebuild_sat(s_recv)
       return False


   rounds_used = 0


   for rnd in range(int(pcfg.max_rounds)):
       rounds_used = rnd + 1
       moved_this_round = 0


       donorsK = np.where(K_cache.astype(float) > float(K_cap) + tol)[0]
       if donorsK.size > 0:
           donorsK = donorsK[np.argsort(-(K_cache[donorsK] - K_cap), kind="mergesort")]
           for s_donor in donorsK.tolist():
               while (K_cache[s_donor] > K_cap + tol) and (moved_this_round < int(pcfg.max_offloads_per_round)):
                   clusters_d = clusters_cache[s_donor]
                   evals_d = evals_cache[s_donor]
                   if clusters_d is None or evals_d is None or len(clusters_d) == 0:
                       break


                   sizes = np.array([int(len(Sb)) for Sb in clusters_d], dtype=int)
                   U_beams = np.array([float(ev.get("U", 0.0)) for ev in evals_d], dtype=float)


                   order = np.lexsort((U_beams, sizes))
                   moved = False
                   for k_beam in order[: min(8, order.size)]:
                       donor_local = np.asarray(clusters_d[int(k_beam)], dtype=int)
                       if donor_local.size == 0:
                           continue
                       if try_move(s_donor, donor_local, mode="K"):
                           moved_this_round += 1
                           moved = True
                           break
                   if not moved:
                       break


       overT = np.maximum(0.0, T_cache - T_cap)
       donorsT = np.where(overT > tol)[0]
       if donorsT.size > 0 and moved_this_round < int(pcfg.max_offloads_per_round):
           donorsT = donorsT[np.argsort(-overT[donorsT], kind="mergesort")]
           for s_donor in donorsT.tolist():
               while (T_cache[s_donor] > T_cap + tol) and (moved_this_round < int(pcfg.max_offloads_per_round)):
                   clusters_d = clusters_cache[s_donor]
                   evals_d = evals_cache[s_donor]
                   if clusters_d is None or evals_d is None or len(clusters_d) == 0:
                       break


                   U_beams = np.array([float(ev.get("U", 0.0)) for ev in evals_d], dtype=float)
                   if U_beams.size == 0:
                       break


                   orderU = np.argsort(-U_beams, kind="mergesort")
                   moved = False
                   for k_beam in orderU[: min(6, orderU.size)]:
                       donor_local = np.asarray(clusters_d[int(k_beam)], dtype=int)
                       if donor_local.size == 0:
                           continue
                       if try_move(s_donor, donor_local, mode="T"):
                           moved_this_round += 1
                           moved = True
                           break
                   if not moved:
                       break


       key_now = state_key()
       if key_now[0] == 0 and key_now[3] == 0:
           Wreq = int(w_min_req_from_Tmax(float(np.max(T_cache)) if S else 0.0))
           if Wreq >= W and moved_this_round < int(pcfg.max_offloads_per_round):
               smooth_budget = min(2, int(pcfg.max_offloads_per_round) - moved_this_round)
               for _ in range(int(max(0, smooth_budget))):
                   s_donor = int(np.argmax(T_cache)) if S else 0
                   clusters_d = clusters_cache[s_donor]
                   evals_d = evals_cache[s_donor]
                   if clusters_d is None or evals_d is None or len(clusters_d) == 0:
                       break
                   U_beams = np.array([float(ev.get("U", 0.0)) for ev in evals_d], dtype=float)
                   if U_beams.size == 0:
                       break
                   k_beam = int(np.argmax(U_beams))
                   donor_local = np.asarray(clusters_d[k_beam], dtype=int)
                   if donor_local.size == 0:
                       break


                   def accept_smooth(kb, ka, Wb, Wa):
                       if ka[0] != 0 or ka[3] != 0:
                           return False
                       return int(Wa) < int(Wb)


                   if try_move(s_donor, donor_local, mode="smooth", accept_predicate=accept_smooth):
                       moved_this_round += 1
                   else:
                       break


       if moved_this_round == 0:
           break


       key_now = state_key()
       if key_now[0] == 0 and key_now[3] == 0:
           Wreq = int(w_min_req_from_Tmax(float(np.max(T_cache)) if S else 0.0))
           if Wreq < W:
               break


   over_T = np.maximum(0.0, T_cache - T_cap)
   over_K = np.maximum(0.0, K_cache.astype(float) - float(K_cap))
   feasible = bool((np.sum(over_T > tol) == 0) and (np.sum(over_K > tol) == 0))


   stats = {
       "enabled": True,
       "J_lanes": J,
       "W_slots": W,
       "Ks_max": K_cap,
       "T_cap": T_cap,
       "K_cap": K_cap,
       "rounds": rounds_used,
       "moves_tried": int(moves_tried),
       "moves_accepted": int(moves_accepted),
       "moves_tried_K": int(moves_tried_K),
       "moves_accepted_K": int(moves_accepted_K),
       "moves_tried_T": int(moves_tried_T),
       "moves_accepted_T": int(moves_accepted_T),
       "moves_tried_smooth": int(moves_tried_smooth),
       "moves_accepted_smooth": int(moves_accepted_smooth),
       "feasible": feasible,
       "n_viol_T": int(np.sum(over_T > tol)),
       "n_viol_K": int(np.sum(over_K > tol)),
       "T_over_sum": float(np.sum(over_T)),
       "K_over_sum": float(np.sum(over_K)),
       "T_over_max": float(np.max(over_T)) if S else 0.0,
       "K_over_max": float(np.max(over_K)) if S else 0.0,
       "T_sum": float(np.sum(T_cache)),
       "T_max": float(np.max(T_cache)) if S else 0.0,
       "K_sum": int(np.sum(K_cache)),
       "K_max": int(np.max(K_cache)) if S else 0,
       "W_min_req": int(w_min_req_from_Tmax(float(np.max(T_cache)) if S else 0.0)),
       "global_cap": float(S) * T_cap,
       "global_impossible": False,
       "T_by_sat": T_cache.copy(),
       "K_by_sat": K_cache.copy(),
   }


   return feasible, sat_user_ids, main_by_sat, stats



@dataclass(frozen=True)
class _Candidate:
   m: int
   assoc: Any
   sat_user_ids: List[np.ndarray]
   pieces_by_sat: dict[int, tuple[Any, list[np.ndarray], list[dict]]]
   payload_stats: dict
   feasible: bool
   assoc_rule: str = "balanced_max_elev"




def _failed_candidate_key(c: _Candidate) -> tuple:
   a = c.payload_stats
   return (
       1 if bool(a.get("global_impossible", False)) else 0,
       int(a.get("n_viol_T", 0)) + int(a.get("n_viol_K", 0)),
       float(a.get("T_over_sum", 0.0)) + float(a.get("K_over_sum", 0.0)),
       float(max(float(a.get("T_over_max", 0.0)), float(a.get("K_over_max", 0.0)))),
       int(c.m),
   )




def _run_method_once(
   users_raw,
   sat_ecef_m: np.ndarray,
   sat_vel_mps: np.ndarray | None,
   cfg: ScenarioConfig,
   local_builder,
   *,
   assoc_rule: str,
   prof: Profiler,
) -> _Candidate:
   ms = cfg.multisat
   pcfg = cfg.payload

   prof.tic("assoc")
   assoc = associate_users_by_rule(
       assoc_rule,
       user_ecef_m=users_raw.ecef_m,
       user_demand_mbps=users_raw.demand_mbps,
       user_qos_w=users_raw.qos_w,
       sat_ecef_m=sat_ecef_m,
       sat_vel_mps=sat_vel_mps,
       elev_mask_deg=ms.elev_mask_deg,
       load_mode=ms.assoc_load_mode,
       slack=ms.assoc_slack,
       max_rounds=ms.assoc_max_rounds,
       max_total_moves=ms.assoc_max_moves,
       seed=int(cfg.run.seed),
   )
   prof.toc("assoc")

   sat_user_ids_init = [np.asarray(x, dtype=int) for x in assoc.sat_user_ids]

   if pcfg.enabled:
       feasible, sat_user_ids_out, pieces_by_sat, payload_stats = _payload_repair_inner(
           users_raw=users_raw,
           sat_ecef_m=sat_ecef_m,
           sat_user_ids_init=sat_user_ids_init,
           cfg=cfg,
           prof=prof,
           local_builder=local_builder,
       )
   else:
       pieces_by_sat = {}
       for s in range(int(sat_ecef_m.shape[0])):
           u, c, e, _T = _build_sat_piece_with_local_builder(users_raw, sat_user_ids_init, sat_ecef_m, s, cfg, prof, local_builder)
           if u is not None and c is not None and e is not None and sat_user_ids_init[s].size > 0:
               pieces_by_sat[s] = (u, c, e)
       feasible = True
       sat_user_ids_out = sat_user_ids_init
       payload_stats = {"enabled": False, "feasible": True}

   return _Candidate(
       m=int(sat_ecef_m.shape[0]),
       assoc=assoc,
       sat_user_ids=sat_user_ids_out,
       pieces_by_sat=pieces_by_sat,
       payload_stats=payload_stats,
       feasible=bool(feasible),
       assoc_rule=str(assoc_rule),
   )



def _run_method_with_prefix_search(
   users_raw,
   sat_ecef_full: np.ndarray,
   sat_vel_full: np.ndarray | None,
   active_sats,
   cfg: ScenarioConfig,
   local_builder,
   *,
   assoc_rule: str,
   prof: Profiler,
) -> tuple[_Candidate | None, float]:
   import time


   pcfg = cfg.payload
   max_prefix = int(pcfg.max_prefix) if pcfg.max_prefix is not None else int(len(active_sats))
   max_prefix = max(1, min(max_prefix, int(len(active_sats))))


   best_feasible: _Candidate | None = None
   best_failed: _Candidate | None = None


   t_start = time.perf_counter()
   for m in range(1, max_prefix + 1):
       sat_ecef_m = sat_ecef_full[:m].copy()
       sat_vel_m = sat_vel_full[:m].copy() if sat_vel_full is not None else None
       cand = _run_method_once(
           users_raw, sat_ecef_m, sat_vel_m, cfg, local_builder,
           assoc_rule=assoc_rule, prof=prof,
       )

       if cand.feasible:
           best_feasible = cand
           break

       if best_failed is None or _failed_candidate_key(cand) < _failed_candidate_key(best_failed):
           best_failed = cand

   runtime_s = time.perf_counter() - t_start
   chosen = best_feasible if best_feasible is not None else best_failed
   return chosen, float(runtime_s)



def _run_method_with_fixed_prefix(
   users_raw,
   sat_ecef_full: np.ndarray,
   sat_vel_full: np.ndarray | None,
   cfg: ScenarioConfig,
   local_builder,
   *,
   m_fixed: int,
   assoc_rule: str,
   prof: Profiler,
) -> tuple[_Candidate, float]:
   import time

   m = int(max(1, min(m_fixed, sat_ecef_full.shape[0])))
   sat_ecef_m = sat_ecef_full[:m].copy()
   sat_vel_m = sat_vel_full[:m].copy() if sat_vel_full is not None else None
   t_start = time.perf_counter()
   cand = _run_method_once(
       users_raw, sat_ecef_m, sat_vel_m, cfg, local_builder,
       assoc_rule=assoc_rule, prof=prof,
   )
   runtime_s = time.perf_counter() - t_start
   return cand, float(runtime_s)




def _candidate_summary(cand: _Candidate | None, cfg: ScenarioConfig) -> dict[str, Any]:
   if cand is None or (not cand.feasible):
       return _nan_summary()
   pieces = list(cand.pieces_by_sat.values())
   return summarize_multisat(pieces, cfg) if pieces else _nan_summary()



def _candidate_meta(cand: _Candidate | None) -> dict[str, Any]:
   if cand is None:
       return {
           "payload_feasible": False,
           "best_m": 0,
           "assoc_moves": 0,
           "n_unserved": 0,
           "n_viol_T": 0,
           "n_viol_K": 0,
           "T_over_sum": 0.0,
           "K_over_sum": 0.0,
       }
   ps = cand.payload_stats or {}
   return {
       "payload_feasible": bool(cand.feasible),
       "best_m": int(cand.m),
       "assoc_moves": int(getattr(cand.assoc, "n_moves", 0)),
       "n_unserved": int(getattr(cand.assoc, "n_unserved", 0)),
       "n_viol_T": int(ps.get("n_viol_T", 0)),
       "n_viol_K": int(ps.get("n_viol_K", 0)),
       "T_over_sum": float(ps.get("T_over_sum", 0.0)),
       "K_over_sum": float(ps.get("K_over_sum", 0.0)),
   }



def run_scenario(cfg: ScenarioConfig) -> dict[str, Any]:
   ms = cfg.multisat
   pcfg = cfg.payload
   verbose = bool(getattr(cfg.run, "verbose", True))


   prof_global = Profiler()
   prof_global.tic("usergen")
   user_list = generate_users(cfg)
   users_raw = pack_users_raw(user_list)
   prof_global.toc("usergen")


   print_config(cfg)


   t0_utc = _parse_time_utc_iso(ms.time_utc_iso) if ms.time_utc_iso else None


   prof_global.tic("sat_select")
   t0_utc, active_sats = sort_active_sats(cfg, t0_utc=t0_utc, n_lat_anchors=3, n_lon_anchors=3, quality_mode="sin")
   prof_global.toc("sat_select")


   if len(active_sats) == 0:
       raise RuntimeError("No active satellites found above elev mask.")


   sat_ecef_full = np.stack([s.ecef_m for s in active_sats], axis=0)
   sat_vel_full = None
   try:
       if all(getattr(s, "ecef_vel_mps", None) is not None for s in active_sats):
           sat_vel_full = np.stack([np.asarray(s.ecef_vel_mps, dtype=float) for s in active_sats], axis=0)
   except Exception:
       sat_vel_full = None


   # Full pipeline for each compared method (existing comparisons keep balanced_max_elev)
   prof_main = Profiler()
   cand_main, rt_main = _run_method_with_prefix_search(users_raw, sat_ecef_full, sat_vel_full, active_sats, cfg, _local_build_main, assoc_rule="balanced_max_elev", prof=prof_main)
   print(f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_main assoc_rule: balanced_max_elev")


   prof_ref = Profiler()
   cand_ref, rt_ref = _run_method_with_prefix_search(users_raw, sat_ecef_full, sat_vel_full, active_sats, cfg, _local_build_main_ref, assoc_rule="balanced_max_elev", prof=prof_ref)
   print(f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_main_ref assoc_rule: balanced_max_elev")


   prof_lb = Profiler()
   cand_lb, rt_lb = _run_method_with_prefix_search(users_raw, sat_ecef_full, sat_vel_full, active_sats, cfg, _local_build_main_ref_lb, assoc_rule="balanced_max_elev", prof=prof_lb)
   print(f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_main_ref_lb assoc_rule: balanced_max_elev")


   # New system-level comparison B:
   # Use the same fixed prefix m chosen by the full proposed method, then compare
   # association rules under the SAME prefix and SAME downstream pipeline.
   sys_fixed_m = int(cand_lb.m) if cand_lb is not None else 1

   prof_sys_pure = Profiler()
   cand_sys_pure, rt_sys_pure = _run_method_with_fixed_prefix(
       users_raw, sat_ecef_full, sat_vel_full, cfg, _local_build_main_ref_lb,
       m_fixed=sys_fixed_m, assoc_rule="pure_max_elev", prof=prof_sys_pure,
   )
   print(f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_main_ref_lb assoc_rule: pure_max_elev fixed_m: {sys_fixed_m}")

   prof_sys_bal = Profiler()
   cand_sys_bal, rt_sys_bal = _run_method_with_fixed_prefix(
       users_raw, sat_ecef_full, sat_vel_full, cfg, _local_build_main_ref_lb,
       m_fixed=sys_fixed_m, assoc_rule="balanced_max_elev", prof=prof_sys_bal,
   )
   print(f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_main_ref_lb assoc_rule: balanced_max_elev fixed_m: {sys_fixed_m}")

   prof_sys_dur = Profiler()
   cand_sys_dur, rt_sys_dur = _run_method_with_fixed_prefix(
       users_raw, sat_ecef_full, sat_vel_full, cfg, _local_build_main_ref_lb,
       m_fixed=sys_fixed_m, assoc_rule="max_service_time", prof=prof_sys_dur,
   )
   print(f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_main_ref_lb assoc_rule: max_service_time fixed_m: {sys_fixed_m}")


   prof_wk_d = Profiler()
   cand_wk_d_rep, rt_wk_d = _run_method_with_prefix_search(users_raw, sat_ecef_full, sat_vel_full, active_sats, cfg, _local_build_wkmeans_repaired(False), assoc_rule="balanced_max_elev", prof=prof_wk_d)
   print(
       f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_wkmeans_repaired use_qos_weight{False} assoc_rule: balanced_max_elev")


   prof_wk_q = Profiler()
   cand_wk_q_rep, rt_wk_q = _run_method_with_prefix_search(users_raw, sat_ecef_full, sat_vel_full, active_sats, cfg, _local_build_wkmeans_repaired(True), assoc_rule="balanced_max_elev", prof=prof_wk_q)
   print(
       f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_wkmeans_repaired use_qos_weight{True} assoc_rule: balanced_max_elev")


   cand_bk_rep = None
   cand_tg_rep = None
   rt_bk = float("nan")
   rt_tg = float("nan")
   if bool(getattr(cfg.run, "enable_fastbp_baselines", True)):
       prof_bk = Profiler()
       cand_bk_rep, rt_bk = _run_method_with_prefix_search(users_raw, sat_ecef_full, sat_vel_full, active_sats, cfg, _local_build_bkmeans_repaired, assoc_rule="balanced_max_elev", prof=prof_bk)
       print(
           f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_bkmeans_repaired assoc_rule: balanced_max_elev")
       prof_tg = Profiler()
       cand_tg_rep, rt_tg = _run_method_with_prefix_search(users_raw, sat_ecef_full, sat_vel_full, active_sats, cfg, _local_build_tgbp_repaired, assoc_rule="balanced_max_elev", prof=prof_tg)
       print(
           f"seed: {cfg.run.seed} users: {cfg.run.n_users} active_sats: {len(active_sats)} method: _local_build_tgbp_repaired assoc_rule: balanced_max_elev")
   else:
       prof_bk = Profiler()
       prof_tg = Profiler()


   main_summary = _candidate_summary(cand_main, cfg)
   ref_summary = _candidate_summary(cand_ref, cfg)
   lb_summary = _candidate_summary(cand_lb, cfg)
   sys_pure_summary = _candidate_summary(cand_sys_pure, cfg)
   sys_balanced_summary = _candidate_summary(cand_sys_bal, cfg)
   sys_service_time_summary = _candidate_summary(cand_sys_dur, cfg)
   wk_demand_rep = _candidate_summary(cand_wk_d_rep, cfg)
   wk_qos_rep = _candidate_summary(cand_wk_q_rep, cfg)
   bk_rep = _candidate_summary(cand_bk_rep, cfg)
   tgbp_rep = _candidate_summary(cand_tg_rep, cfg)


   wk_demand_fixed = _nan_summary()
   wk_qos_fixed = _nan_summary()
   bk_fixed = _nan_summary()
   tgbp_fixed = _nan_summary()


   chosen_top = cand_lb if (cand_lb is not None and cand_lb.feasible) else (cand_ref if (cand_ref is not None and cand_ref.feasible) else cand_main)
   payload_feasible_top = bool(chosen_top is not None and chosen_top.feasible)
   m_used = int(chosen_top.m) if chosen_top is not None else 0


   if verbose:
       print_summary("Main algorithm (payload-feasible) [global]", main_summary, cfg)
       print_summary("Main + enterprise refinement (payload-feasible) [global]", ref_summary, cfg)
       print_summary("Main + enterprise + load-balance (payload-feasible) [global]", lb_summary, cfg)
       print_summary(f"System compare: pure max-elevation @ fixed prefix m={sys_fixed_m}", sys_pure_summary, cfg)
       print_summary(f"System compare: balanced max-elevation @ fixed prefix m={sys_fixed_m}", sys_balanced_summary, cfg)
       print_summary(f"System compare: max-service-time @ fixed prefix m={sys_fixed_m}", sys_service_time_summary, cfg)
       print_summary("WKMeans++ baseline (demand) after payload certification [global]", wk_demand_rep, cfg)
       print_summary("WKMeans++ baseline (demand*qos) after payload certification [global]", wk_qos_rep, cfg)
       if bool(getattr(cfg.run, "enable_fastbp_baselines", True)):
           print_summary("BK-Means baseline (payload-certified repaired) [global]", bk_rep, cfg)
           print_summary("TGBP baseline (payload-certified repaired) [global]", tgbp_rep, cfg)


   time_split_s = float(rt_main) if (cand_main is not None and cand_main.feasible) else float("nan")
   time_ent_ref_s = float(max(0.0, rt_ref - rt_main)) if (cand_ref is not None and cand_ref.feasible and cand_main is not None and cand_main.feasible) else float("nan")
   time_lb_ref_s = float(max(0.0, rt_lb - rt_ref)) if (cand_lb is not None and cand_lb.feasible and cand_ref is not None and cand_ref.feasible) else float("nan")


   ent_prof = prof_lb if (cand_lb is not None and cand_lb.feasible) else prof_ref


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
       "theta_3db_deg": cfg.beam.theta_3db_deg,


       "ms_tle_path": ms.tle_path,
       "ms_time_utc": t0_utc.isoformat(),
       "ms_elev_mask_deg": ms.elev_mask_deg,
       "ms_n_active": int(m_used),
       "ms_n_unserved": int(getattr(chosen_top.assoc, "n_unserved", 0)) if chosen_top is not None else 0,
       "ms_assoc_moves": int(getattr(chosen_top.assoc, "n_moves", 0)) if chosen_top is not None else 0,


       "payload_enabled": bool(pcfg.enabled),
       "payload_J_lanes": float(pcfg.J_lanes),
       "payload_W_slots": int(pcfg.W_slots),
       "payload_Ks_max": int(pcfg.Ks_max),


       "payload_moves_tried": int(chosen_top.payload_stats.get("moves_tried", 0)) if chosen_top is not None else 0,
       "payload_moves_accepted": int(chosen_top.payload_stats.get("moves_accepted", 0)) if chosen_top is not None else 0,
       "payload_moves_tried_K": int(chosen_top.payload_stats.get("moves_tried_K", 0)) if chosen_top is not None else 0,
       "payload_moves_accepted_K": int(chosen_top.payload_stats.get("moves_accepted_K", 0)) if chosen_top is not None else 0,
       "payload_moves_tried_T": int(chosen_top.payload_stats.get("moves_tried_T", 0)) if chosen_top is not None else 0,
       "payload_moves_accepted_T": int(chosen_top.payload_stats.get("moves_accepted_T", 0)) if chosen_top is not None else 0,
       "payload_moves_tried_smooth": int(chosen_top.payload_stats.get("moves_tried_smooth", 0)) if chosen_top is not None else 0,
       "payload_moves_accepted_smooth": int(chosen_top.payload_stats.get("moves_accepted_smooth", 0)) if chosen_top is not None else 0,


       "payload_feasible": bool(payload_feasible_top),
       "payload_best_m": int(m_used),


       "payload_T_cap": float(chosen_top.payload_stats.get("T_cap", float(pcfg.J_lanes) * float(pcfg.W_slots))) if chosen_top is not None else float(pcfg.J_lanes) * float(pcfg.W_slots),
       "payload_K_cap": int(chosen_top.payload_stats.get("K_cap", int(pcfg.Ks_max))) if chosen_top is not None else int(pcfg.Ks_max),


       "payload_n_viol_T": int(chosen_top.payload_stats.get("n_viol_T", 0)) if chosen_top is not None else 0,
       "payload_n_viol_K": int(chosen_top.payload_stats.get("n_viol_K", 0)) if chosen_top is not None else 0,
       "payload_T_over_sum": float(chosen_top.payload_stats.get("T_over_sum", 0.0)) if chosen_top is not None else 0.0,
       "payload_K_over_sum": float(chosen_top.payload_stats.get("K_over_sum", 0.0)) if chosen_top is not None else 0.0,
       "payload_T_over_max": float(chosen_top.payload_stats.get("T_over_max", 0.0)) if chosen_top is not None else 0.0,
       "payload_K_over_max": float(chosen_top.payload_stats.get("K_over_max", 0.0)) if chosen_top is not None else 0.0,
       "payload_T_sum": float(chosen_top.payload_stats.get("T_sum", 0.0)) if chosen_top is not None else 0.0,
       "payload_T_max": float(chosen_top.payload_stats.get("T_max", 0.0)) if chosen_top is not None else 0.0,
       "payload_K_sum": int(chosen_top.payload_stats.get("K_sum", 0)) if chosen_top is not None else 0,
       "payload_K_max": int(chosen_top.payload_stats.get("K_max", 0)) if chosen_top is not None else 0,
       "payload_W_min_req": int(chosen_top.payload_stats.get("W_min_req", 0)) if chosen_top is not None else 0,
       "payload_global_cap": float(chosen_top.payload_stats.get("global_cap", 0.0)) if chosen_top is not None else 0.0,
       "payload_global_impossible": bool(chosen_top.payload_stats.get("global_impossible", False)) if chosen_top is not None else False,


       "time_usergen_s": prof_global.t.get("usergen", 0.0),
       "time_sat_select_s": prof_global.t.get("sat_select", 0.0),
       "time_assoc_s": prof_lb.t.get("assoc", prof_ref.t.get("assoc", prof_main.t.get("assoc", 0.0))),
       "time_split_s": time_split_s,
       "time_ent_ref_s": time_ent_ref_s,
       "time_lb_ref_s": time_lb_ref_s,
       "time_sys_pure_max_elev_s": float(rt_sys_pure),
       "time_sys_balanced_max_elev_s": float(rt_sys_bal),
       "time_sys_max_service_time_s": float(rt_sys_dur),


       "eval_calls": int((prof_lb if (cand_lb is not None and cand_lb.feasible) else prof_main).c.get("eval_calls", 0)),
       "n_splits": int((prof_lb if (cand_lb is not None and cand_lb.feasible) else prof_main).c.get("n_splits", 0)),
       "ent_moves_tried": int(ent_prof.c.get("ent_moves_tried", 0)),
       "ent_moves_accepted": int(ent_prof.c.get("ent_moves_accepted", 0)),
       "lb_moves_tried": int(prof_lb.c.get("lb_moves_tried", 0)),
       "lb_moves_accepted": int(prof_lb.c.get("lb_moves_accepted", 0)),


       "time_baseline_without_qos_s": float(rt_wk_d) if (cand_wk_d_rep is not None and cand_wk_d_rep.feasible) else float("nan"),
       "time_baseline_with_qos_s": float(rt_wk_q) if (cand_wk_q_rep is not None and cand_wk_q_rep.feasible) else float("nan"),
       "time_baseline_bkmeans_s": float(rt_bk) if (cand_bk_rep is not None and cand_bk_rep.feasible) else float("nan"),
       "time_baseline_tgbp_s": float(rt_tg) if (cand_tg_rep is not None and cand_tg_rep.feasible) else float("nan"),


       # Method-level payload metadata
       "main_payload_feasible": bool(cand_main is not None and cand_main.feasible),
       "main_best_m": int(cand_main.m) if cand_main is not None else 0,
       "main_ref_payload_feasible": bool(cand_ref is not None and cand_ref.feasible),
       "main_ref_best_m": int(cand_ref.m) if cand_ref is not None else 0,
       "main_ref_lb_payload_feasible": bool(cand_lb is not None and cand_lb.feasible),
       "main_ref_lb_best_m": int(cand_lb.m) if cand_lb is not None else 0,
       "wk_demand_rep_payload_feasible": bool(cand_wk_d_rep is not None and cand_wk_d_rep.feasible),
       "wk_demand_rep_best_m": int(cand_wk_d_rep.m) if cand_wk_d_rep is not None else 0,
       "wk_qos_rep_payload_feasible": bool(cand_wk_q_rep is not None and cand_wk_q_rep.feasible),
       "wk_qos_rep_best_m": int(cand_wk_q_rep.m) if cand_wk_q_rep is not None else 0,
       "bk_rep_payload_feasible": bool(cand_bk_rep is not None and cand_bk_rep.feasible),
       "bk_rep_best_m": int(cand_bk_rep.m) if cand_bk_rep is not None else 0,
       "tgbp_rep_payload_feasible": bool(cand_tg_rep is not None and cand_tg_rep.feasible),
       "tgbp_rep_best_m": int(cand_tg_rep.m) if cand_tg_rep is not None else 0,

       # New system-level fixed-prefix association comparison metadata
       "sys_assoc_fixed_prefix_m": int(sys_fixed_m),
       "sys_pure_max_elev_payload_feasible": bool(cand_sys_pure is not None and cand_sys_pure.feasible),
       "sys_pure_max_elev_best_m": int(cand_sys_pure.m) if cand_sys_pure is not None else 0,
       "sys_pure_max_elev_assoc_moves": int(getattr(cand_sys_pure.assoc, "n_moves", 0)) if cand_sys_pure is not None else 0,
       "sys_pure_max_elev_n_unserved": int(getattr(cand_sys_pure.assoc, "n_unserved", 0)) if cand_sys_pure is not None else 0,
       "sys_balanced_max_elev_payload_feasible": bool(cand_sys_bal is not None and cand_sys_bal.feasible),
       "sys_balanced_max_elev_best_m": int(cand_sys_bal.m) if cand_sys_bal is not None else 0,
       "sys_balanced_max_elev_assoc_moves": int(getattr(cand_sys_bal.assoc, "n_moves", 0)) if cand_sys_bal is not None else 0,
       "sys_balanced_max_elev_n_unserved": int(getattr(cand_sys_bal.assoc, "n_unserved", 0)) if cand_sys_bal is not None else 0,
       "sys_max_service_time_payload_feasible": bool(cand_sys_dur is not None and cand_sys_dur.feasible),
       "sys_max_service_time_best_m": int(cand_sys_dur.m) if cand_sys_dur is not None else 0,
       "sys_max_service_time_assoc_moves": int(getattr(cand_sys_dur.assoc, "n_moves", 0)) if cand_sys_dur is not None else 0,
       "sys_max_service_time_n_unserved": int(getattr(cand_sys_dur.assoc, "n_unserved", 0)) if cand_sys_dur is not None else 0,

       "main": main_summary,
       "main_ref": ref_summary,
       "main_ref_lb": lb_summary,
       "sys_pure_max_elev": sys_pure_summary,
       "sys_balanced_max_elev": sys_balanced_summary,
       "sys_max_service_time": sys_service_time_summary,
       "wk_demand_fixed": wk_demand_fixed,
       "wk_demand_rep": wk_demand_rep,
       "wk_qos_fixed": wk_qos_fixed,
       "wk_qos_rep": wk_qos_rep,
       "bk_fixed": bk_fixed,
       "bk_rep": bk_rep,
       "tgbp_fixed": tgbp_fixed,
       "tgbp_rep": tgbp_rep,
   }
