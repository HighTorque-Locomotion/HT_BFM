## 2026-09-02 12:04 CST - Add lateral and yaw expert motions to Stage2

- Context: PiPlus Stage2 command encoder, resume from checkpoint_18700.pt on GPU11 GPU1.
- Phenomenon: The previous expert set had only forward, stand, and backward motions; turn commands had no matching lateral/yaw expert AMP motions. The prior AMP=0.08 challenger also destabilized PPO, so the original checkpoint was preserved.
- Analysis: `dataset/pkl_cmd/piplus_cmd_balanced_amp_9.pkl` contains nine PiPlus motion entries with fixed body-frame commands: stand, backward, forward slow/medium/fast, lateral left/right, and yaw left/right. The existing `PiPlusAMPExpertDataset.from_pkl` contract accepts this mapping and produces the same 202-D AMP features.
- Adjustment: Switch expert dataset from the 5-motion `piplus_lse_balanced_forward_turn_stand_backward.pkl` to `piplus_cmd_balanced_amp_9.pkl`; command stand probability `0.10 -> 0.25`; turn probability `0.15 -> 0.20`. Keep AMP weight `0.04`, discriminator LR `1e-5`, policy LR `1e-5`, and checkpoint `18700`.
- Rationale: Provide direct AMP style coverage for lateral and yaw responses while increasing explicit stand samples; conservative optimizer/AMP values avoid the previously observed discriminator saturation and value-loss spikes.
- Expected effect: Better lateral/yaw command response, more expert-like contact/gait style, and lower motion under zero command. Monitor per-bin MAE/correlation, AMP score/gap, stand metrics, PPO update fraction, value loss, and terminations.
- Result: Pending mature checkpoint window.
- Files/commands: remote run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_balanced9_stand25_turn20_amp004_resume18700_20260902`; tmux `bfm_balanced9_stand25_turn20_amp004_resume18700_20260902`; source checkpoint `logs/amp_stage2_piplus_lse_4gpu_4096env_1m_yaw_amp_resume18400_20260803/checkpoint_18700.pt`.

## 2026-09-02 12:08 CST - Early nine-motion validation

- Context: Same nine-motion Stage2 challenger after independent SSH reconnect.
- Phenomenon: The run advanced from iteration 18704 to 18725 without a numerical or termination failure.
- Analysis: Early tracking is comparable to or better than the five-motion baseline (vx MAE 0.147, vy MAE 0.119, planar MAE 0.207); AMP score remains positive at 0.285 but the window is too short for a style verdict. Stand contribution is nonzero because explicit stand sampling is 25%.
- Adjustment: None; keep one mutation active and wait for checkpoint-aligned evidence.
- Rationale: Avoid confounding the new lateral/yaw expert coverage with another reward or optimizer change.
- Expected effect: Maintain zero falls/low terminations while lateral/yaw bins and AMP discriminator gap stabilize.
- Result: Challenger pending; checkpoint_18800 not yet written.
- Files/commands: tmux `bfm_balanced9_stand25_turn20_amp004_resume18700_20260902`; remote console `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_balanced9_stand25_turn20_amp004_resume18700_20260902/console.log`.

## 2026-09-02 12:12 CST - Nine-motion run remains checkpoint-pending

- Context: Same nine-motion challenger, independently checked after SSH reconnect.
- Phenomenon: The process remains alive at iteration 18764 with no new checkpoint yet; the latest row has zero termination/fall/truncation.
- Analysis: Tracking is currently vx MAE 0.153, vy MAE 0.115, yaw MAE 0.203, with vx/yaw correlation 0.953/0.868 and AMP score 0.167. This is healthy operationally but not enough to claim a mature style improvement.
- Adjustment: None.
- Rationale: Keep the turn/lateral expert-data mutation isolated until a checkpoint-aligned window exists.
- Expected effect: Preserve stand safety while the discriminator adapts to the nine-motion expert distribution.
- Result: Challenger pending; `checkpoint_18800.pt` still absent.
- Files/commands: remote run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_balanced9_stand25_turn20_amp004_resume18700_20260902`; tmux `bfm_balanced9_stand25_turn20_amp004_resume18700_20260902`.

## 2026-09-02 12:14 CST - Conservative restart after nine-motion instability

- Context: Same PiPlus Stage2 task, original checkpoint_18700 preserved.
- Phenomenon: The stand25/turn20 nine-motion challenger degraded by iteration 18772 (yaw correlation 0.485, yaw MAE 0.353, value loss 2.90, PPO update fraction 0.088), so it was stopped before producing a new checkpoint.
- Analysis: The data shift plus stale PPO/discriminator optimizer state was too aggressive for the first challenger. The 18700 checkpoint remains the rollback point.
- Adjustment: Restarted from 18700 with the same nine-motion expert dataset; policy/discriminator LR `1e-5 -> 5e-6`, AMP weight `0.04 -> 0.03`, PPO epochs `4 -> 3`, target KL `0.025 -> 0.01`, stand/turn sampling `0.25/0.20 -> 0.20/0.20`.
- Rationale: Reduce update magnitude and adversarial pressure while retaining explicit lateral/yaw expert coverage and stand supervision.
- Expected effect: Higher PPO update completion, lower value-loss spikes, stable standing, and gradual improvement in all command bins and AMP score.
- Result: Fresh SSH verification shows PID 2514437 alive at iteration 18706; termination/fall/truncation are zero, AMP score 0.559, PPO update fraction 0.068 on the first observed row. Mature checkpoint evaluation is pending.
- Files/commands: remote run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_balanced9_stand20_turn20_amp003_lr5e6_resume18700_20260902`; tmux `bfm_balanced9_stand20_turn20_amp003_lr5e6_resume18700_20260902`.

## 2026-09-02 12:22 CST - Reset discriminator state for nine-motion distribution

- Context: Recovered from the repeated yaw-collapse signature in two nine-motion challengers.
- Phenomenon: Directly loading the old five-motion discriminator against nine motions produced discriminator saturation and yaw tracking collapse before checkpoint_18800.
- Analysis: The policy/value state in checkpoint_18700 is compatible, but the discriminator state is distribution-specific. A fresh 202-D discriminator removes that stale-state confound while preserving the requested Stage2 policy resume.
- Adjustment: Derived remote `checkpoint_18700_discreset9.pt` from `checkpoint_18700.pt`: replaced only `discriminator` with a fresh `AMPDiscriminator(202)` state and removed `discriminator_optimizer`; resumed with nine-motion expert data, stand/turn `0.20/0.20`, AMP `0.03`, policy LR `5e-6`, discriminator LR `1e-5`, PPO `3` epochs, target KL `0.01`.
- Rationale: Let the adversary fit the new lateral/yaw expert distribution from a neutral state instead of forcing old five-motion discriminator weights to reinterpret it.
- Expected effect: Avoid early yaw collapse, keep PPO/value updates finite, and retain stand/turn expert coverage.
- Result: Independent SSH verification confirms PID 2516429 alive at iteration 18717; discriminator gap `0.020`, AMP score `-0.116`, vx/yaw MAE `0.149/0.207`, vx/yaw correlation `0.948/0.868`, termination/fall/truncation zero. Still pending mature checkpoint.
- Files/commands: remote derived checkpoint `.../checkpoint_18700_discreset9.pt`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_balanced9_discreset9_stand20_turn20_amp003_lr5e6_resume18700_20260902`; tmux `bfm_balanced9_discreset9_stand20_turn20_amp003_lr5e6_resume18700_20260902`.

## 2026-09-02 12:28 CST - Weighted turn expert restart

- Context: The discriminator-reset nine-motion run repeated yaw collapse by iteration 18728, so it was stopped before checkpoint output.
- Phenomenon: Equal weighting of four generated lateral/yaw command motions against five real LSE motions remained too disruptive despite fresh discriminator state.
- Analysis: Preserve the original five-motion gait prior and add turn/lateral style at half relative weight by duplicating each original motion twice; keep the policy state from 18700 and reset the discriminator again.
- Adjustment: Created remote `dataset/pkl_cmd/piplus_lse_balanced5_plus_turn4_weighted.pkl` with 14 entries (10 original duplicates + 4 lateral/yaw entries), and `checkpoint_18700_discreset14.pt` with a fresh discriminator. Started conservative PPO (`5e-6`, 3 epochs, target KL `0.01`), discriminator LR `1e-5`, AMP `0.03`, stand/turn `0.20/0.20`.
- Rationale: Add requested turn coverage without letting generated turn motions dominate the expert style distribution.
- Expected effect: Retain stable forward/backward/stand gait while gradually improving lateral/yaw response and zero-command standing.
- Result: Independent SSH verification confirms PID 2518470 alive at iteration 18711; termination/fall/truncation zero, vx/yaw MAE `0.145/0.207`, vx/yaw correlation `0.954/0.882`, discriminator gap `0.024`, PPO update fraction `0.125`. Mature checkpoint pending.
- Files/commands: remote dataset `dataset/pkl_cmd/piplus_lse_balanced5_plus_turn4_weighted.pkl`; derived checkpoint `.../checkpoint_18700_discreset14.pt`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_discreset_stand20_turn20_amp003_lr5e6_resume18700_20260902`; tmux `bfm_weighted14_discreset_stand20_turn20_amp003_lr5e6_resume18700_20260902`.

## 2026-09-02 12:35 CST - Reset policy and discriminator optimizers

- Context: Weighted14 discriminator-reset run repeated the same yaw collapse around iteration 18728 despite conservative scalar settings.
- Phenomenon: The repeated collapse implicated stale policy Adam moments in the 18700 checkpoint in addition to the stale discriminator state.
- Analysis: Keep the 18700 policy/value tensors as the behavioral base, but start both optimizers fresh so old momentum cannot push the command encoder away during the new expert-distribution transition.
- Adjustment: Derived `checkpoint_18700_optreset14.pt` by removing `policy_optimizer` and replacing `discriminator` with a fresh 202-D discriminator; removed `discriminator_optimizer`. Resumed with weighted14 expert set, policy LR `5e-6`, discriminator LR `1e-5`, AMP `0.03`, PPO 3 epochs, target KL `0.01`, stand/turn `0.20/0.20`.
- Rationale: Isolate the policy weights from stale optimizer dynamics while retaining the requested 18700 resume lineage and added turn/lateral expert coverage.
- Expected effect: Prevent the deterministic early yaw collapse, keep value loss finite, and maintain zero-command standing.
- Result: Independent SSH verification confirms PID 2520355 alive at iterations 18702-18703 with termination/fall/truncation zero, vx/yaw MAE `0.140/0.197`, vx/yaw correlation `0.966/0.877`, and value loss `0.26-0.36`. PPO update fraction is low during the fresh-optimizer warmup; mature checkpoint pending.
- Files/commands: remote derived checkpoint `.../checkpoint_18700_optreset14.pt`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_optreset_stand20_turn20_amp003_lr5e6_resume18700_20260902`; tmux `bfm_weighted14_optreset_stand20_turn20_amp003_lr5e6_resume18700_20260902`.

## 2026-09-02 13:08 CST - Launch repair blocked by stale GPU1 context

- Context: Smoothness challenger using weighted14 expert data, fresh optimizers, policy LR `2e-6`, command smoothing `0.05`.
- Phenomenon: Isaac initialization repeatedly stops after GLFW warnings with no iteration output; selected PID was terminated. GPU1 still reports two orphaned CUDA contexts using about 15.3 GiB, while GPU0/GPU2 are occupied by unrelated jobs.
- Analysis: This is an external GPU/Isaac launch blocker, not a policy checkpoint failure. The same checkpoint and command were preserved for retry; no other GPU was touched.
- Adjustment: None to training semantics. Launch repair used `DISPLAY=` and a unique `UFO_STAGE2_CACHE_DIR`, then stopped the stuck session after no progress.
- Rationale: Avoid changing policy/data while the authorized GPU1 resource is unavailable.
- Expected effect: Retry unchanged smoothness challenger once GPU1 stale contexts clear.
- Result: No active selected training process; original `checkpoint_18700.pt` and `checkpoint_18700_optreset14.pt` remain intact.
- Files/commands: failed run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_optreset_smooth005_lr2e6_stand20_turn20_resume18700_20260902`; tmux `bfm_weighted14_optreset_smooth005_lr2e6_resume18700_20260902`.

## 2026-09-02 13:14 CST - Smoothness challenger launch repaired

- Context: Same weighted14/optreset14 challenger with command smoothing `0.05` and policy LR `2e-6`.
- Phenomenon: First launch was stuck in Isaac GLFW initialization because two old selected challengers still held GPU1; both were explicitly cleaned up, and GPU1 memory returned to 2 MiB.
- Analysis: The failure was supervisor/resource cleanup, not training behavior. Reusing the same checkpoint and parameters after clearing the selected stale sessions preserves attribution.
- Adjustment: Relaunched with `DISPLAY=`, `MUJOCO_EGL_DEVICE_ID=0`, and unique `UFO_STAGE2_CACHE_DIR=/tmp/ufo_amp_stage2_cache_smooth005_retry2`; no training semantics changed.
- Rationale: Isolate Isaac cache/display state and verify persistence after SSH disconnect.
- Expected effect: Complete headless initialization and collect stable smoothness/tracking metrics.
- Result: Independent SSH verification confirms PID 2529731 alive at iterations 18701-18703; vx/yaw MAE `0.118/0.223`, then `0.127/0.195`, then `0.144/0.206`; termination/fall/truncation zero. Fresh optimizer and weighted14 config loaded successfully.
- Files/commands: run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_optreset_smooth005_lr2e6_stand20_turn20_resume18700_20260902`; tmux `bfm_weighted14_optreset_smooth005_lr2e6_resume18700_20260902`; retry log `console_retry2.log`.

## 2026-09-02 13:32 CST - Smooth005 challenger remains viable

- Context: Same weighted14, fresh policy/discriminator optimizer, command smoothing `0.05`, policy LR `2e-6`.
- Phenomenon: After launch repair and GPU1 cleanup, the run reached iteration 18729 with no sustained termination/fall failure and no checkpoint yet.
- Analysis: The last 20-row range narrowed relative to the prior smoothing `0.15` run: yaw correlation stayed `0.66-0.89`, yaw MAE `0.187-0.299`, vx correlation `0.72-0.96`, and value loss `0.16-1.33`; the latest row recovered to yaw MAE `0.252`, value loss `0.70`.
- Adjustment: None; keep the challenger running to the first checkpoint-aligned window.
- Rationale: The lower command-transition gain appears to reduce the severe periodic spikes, but maturity is not established until checkpoint_18800.
- Expected effect: Better transition smoothness and stable stand/turn response without sacrificing the weighted expert style prior.
- Result: Challenger pending; process PID 2529731 remains live after independent SSH reconnect.
- Files/commands: remote run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_optreset_smooth005_lr2e6_stand20_turn20_resume18700_20260902`; tmux `bfm_weighted14_optreset_smooth005_lr2e6_resume18700_20260902`.

## 2026-09-02 12:47 CST - Transient rollout spike recovered

- Context: Fresh-optimizer weighted14 challenger, PID 2520355.
- Phenomenon: Iteration 18731 showed a one-row vx/yaw tracking spike and value loss 7.17, then iterations 18732-18735 recovered without termination/fall.
- Analysis: The event is transient rollout variance rather than persistent policy collapse; discriminator gap remains near zero and subsequent rows return to vx/yaw MAE `0.157/0.220` or better.
- Adjustment: None; continue the same run and judge by checkpoint-aligned windows rather than a single row.
- Rationale: Avoid unnecessary restart after verified recovery.
- Expected effect: Stable aggregate tracking and stand behavior as the fresh optimizers warm up.
- Result: Challenger remains running at iteration 18735; `checkpoint_18800` pending.
- Files/commands: remote console `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_optreset_stand20_turn20_amp003_lr5e6_resume18700_20260902/console.log`.

## 2026-09-02 13:47 CST - Resume from mature checkpoint 18900

- Context: Smooth005 weighted14 challenger; prior run reached checkpoint_18900 before evaluation-induced Isaac GPU warnings.
- Phenomenon: The prior training process stopped after the evaluation phase, while checkpoint_18900 remained complete and GPU1 was subsequently released.
- Analysis: Resume from the latest complete 18900 state to retain learned policy/discriminator/optimizer progress; avoid concurrent player launches during training.
- Adjustment: Restarted unchanged smooth005 weighted14 configuration from `checkpoint_18900.pt` with `DISPLAY=`, isolated cache, GPU1; no reward or architecture change.
- Rationale: Continue the validated training lineage while keeping evaluation resource-isolated.
- Expected effect: Stable continuation toward checkpoint_19000 with no Isaac/GLFW interference.
- Result: Independent SSH verification at iteration 18925 shows vx/yaw MAE `0.154/0.218`, vx/yaw correlation `0.946/0.843`, value loss `0.101`, termination/fall/truncation zero; checkpoint_18900 remains the rollback point.
- Files/commands: run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_optreset_smooth005_lr2e6_resume18900_20260902`; tmux `bfm_weighted14_optreset_smooth005_lr2e6_resume18900_20260902`.

## 2026-09-02 13:28 CST - First weighted14 checkpoint screen

- Context: Smooth005 weighted14 challenger, checkpoint_18800.
- Phenomenon: A complete `checkpoint_18800.pt` was written and the process continued to iteration 18802/18860 without a fatal error.
- Analysis: Remote state validation confirms policy, discriminator, both optimizers, AMP normalizer, iteration 18800, expert_motion_count 14, and command_smoothing 0.05. The 18802 row has vx/yaw MAE `0.158/0.215`, vx/yaw correlation `0.951/0.851`, value loss `0.106`, termination `7.6e-6`.
- Adjustment: None; keep this challenger running for a longer mature window.
- Rationale: The first checkpoint establishes resumability but not promotion; fixed-command CPU MuJoCo screening showed noisy instantaneous velocities, so no behavior claim is made yet.
- Expected effect: Retain low termination and improve all command bins and AMP style as the run matures.
- Result: Challenger pending; checkpoint_18800 is valid and playable, but no promotion decision.
- Files/commands: checkpoint `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_optreset_smooth005_lr2e6_stand20_turn20_resume18700_20260902/checkpoint_18800.pt`; fixed screens `/tmp/eval18800_{forward,backward,lateral_left,yaw_left}.log`.

## 2026-09-02 14:31 CST - Reject stand action gate experiment and restore baseline code

- Context: checkpoint_19000, weighted14 Stage2 continuation; user goal requires static zero-command standing.
- Phenomenon: Fixed-command MuJoCo screening showed severe drift under stand even before the patch, so a stand action gate was tested. Zero-action gating was unstable; a frozen-BFM backward-map replacement still drifted and raised early termination/contact penalties in training.
- Analysis: The MuJoCo playback path does not establish a valid Isaac-trained balance verdict; the BFM backward-map action is not a verified standing controller for this checkpoint. The gate also changes rollout/action credit semantics.
- Adjustment: Rejected and reverted the stand-gate code experiment; restored remote and local tracked source files to exact `08bed01c548239589bb4764052e0b7c87cc3cfed` content. Preserved checkpoints 18800/18900/19000 and all dataset artifacts.
- Rationale: Do not ship an unvalidated action-routing change that harms Isaac training while the evaluator has known simulator mismatch.
- Expected effect: Resume the validated Stage2 action path; future static-standing work requires an Isaac-native standing metric/controller or a verified BFM standing latent, not an unvalidated MuJoCo gate.
- Result: Failed stand-gate challenger stopped; baseline code restored and 20 remote unit tests passed. `checkpoint_19000.pt` remains the latest compatible training checkpoint.
- Files/commands: remote backup `/tmp/ht_bfm_backup_stand_bfm_gate_20260902`; rejected run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_standgate_bfm_weighted14_smooth005_lr2e6_resume19000_retry2_20260902`.

## 2026-09-02 14:45 CST - Resume baseline after rejected gate

- Context: Restored exact 08bed01 Stage2 action/PPO path after stand-gate rejection.
- Phenomenon: Baseline resumed from complete checkpoint_19000 with weighted14 expert data and smooth005 settings.
- Analysis: Early iteration 19031 had a transient tracking spike, but iterations 19038-19043 recovered to vx/yaw MAE `0.146-0.182/0.212-0.286`, vx/yaw correlation `0.865-0.954/0.624-0.849`, value loss `0.064-0.67`, and zero termination/fall.
- Adjustment: No new training parameter; restored baseline action path and resumed on GPU1.
- Rationale: Preserve the validated checkpoint lineage while avoiding unvalidated static-action routing.
- Expected effect: Continue stable Stage2 learning; future static-standing changes require Isaac-native evidence.
- Result: PID 2558669 remains live at iteration 19043; checkpoint_19100 pending.
- Files/commands: run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_baseline_resume19000_20260902`; tmux `bfm_weighted14_baseline_resume19000_20260902`.

## 2026-09-02 14:55 CST - Baseline checkpoint 19100 verified

- Context: Exact 08bed01 action/PPO path, weighted14 expert data, resumed from checkpoint_19000.
- Phenomenon: The run wrote complete `checkpoint_19100.pt` and remains live at iteration 19100 after independent SSH reconnect.
- Analysis: Latest row has vx/yaw MAE `0.153/0.228`, vx/yaw correlation `0.947/0.832`, vx/yaw response slope `0.666/0.872`, value loss `0.083`, and zero termination/fall/truncation. AMP score is `-0.176`, so style alignment remains unresolved despite stable tracking.
- Adjustment: None; keep the baseline lineage running and do not evaluate concurrently on the Isaac GPU.
- Rationale: Preserve a stable checkpoint while separating the known MuJoCo/Isaac evaluation mismatch from training behavior.
- Expected effect: Continue collecting a mature post-19100 window before any further architecture change.
- Result: `checkpoint_19100.pt` is the latest verified rollback point; process remains active.
- Files/commands: run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_baseline_resume19000_20260902`; checkpoint `checkpoint_19100.pt`; tmux `bfm_weighted14_baseline_resume19000_20260902`.

## 2026-09-02 15:10 CST - Enable command-matched AMP expert sampling

- Context: Baseline weighted14 resumed from checkpoint_19100; repeated rollout/value outliers and unconditioned AMP sampling were observed.
- Phenomenon: The 202-D discriminator sampled arbitrary expert windows regardless of the current stand/forward/backward/lateral/turn command, so its style gradient was not direction-aligned.
- Analysis: Add category-aware expert sampling without changing discriminator dimensions or policy interface. Command categories map to pools built from motion names/command metadata; missing pools fall back to the global expert pool.
- Adjustment: Added `--amp-command-matched` and category pools (`stand`, `forward`, `backward`, `lateral`, `turn`); resumed from checkpoint_19100 with this flag, unchanged weighted14 data and smooth005/PPO parameters.
- Rationale: Align AMP style comparisons with the commanded behavior while preserving checkpoint compatibility.
- Expected effect: More meaningful directional AMP gradients, fewer command/style conflicts, and better expert-like gait across turns and stand.
- Result: Independent SSH verification confirms PID 2566681 alive at iterations 19107-19109; vx/yaw MAE `0.150-0.189/0.227-0.271`, vx/yaw correlation `0.874-0.950/0.785-0.828`, value loss `0.066-0.098`, termination/fall zero. Challenger pending mature checkpoint.
- Files/commands: modified `humanoidverse/amp_stage2.py`, `tests/test_amp_stage2.py`; remote backup `/tmp/ht_bfm_backup_amp_cmdmatched_20260902`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_ampcmdmatched_resume19100_20260902`; tmux `bfm_ampcmdmatched_resume19100_20260902`.

## 2026-09-02 15:06 CST - Start 30-minute remote monitoring loop

- Context: Command-matched AMP challenger PID 2566681 on GPU1.
- Phenomenon: User requested monitoring once every 30 minutes instead of per-turn polling.
- Analysis: A read-only tmux loop can record process identity, GPU memory, latest JSON metric row, and newest checkpoint without touching training state.
- Adjustment: Started tmux `bfm_ampcmdmatched_monitor_30m`, logging every 1800 seconds to `/tmp/bfm_ampcmdmatched_monitor_30m.log`; corrected the loop to `cd` into the project directory.
- Rationale: Keep long-running supervision persistent across SSH disconnects while preserving the selected training lineage.
- Expected effect: Detect stalled process, GPU loss, checkpoint progression, or metric degradation at the requested cadence.
- Result: Initial monitor sample confirms PID 2566681 alive, GPU1 ~15.5 GiB, current iteration 19190; no checkpoint_19200 yet.
- Files/commands: remote monitor log `/tmp/bfm_ampcmdmatched_monitor_30m.log`; selected run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_ampcmdmatched_resume19100_20260902`.

## 2026-09-02 15:24 CST - Half-hour monitor confirms checkpoint 19200

- Context: Command-matched AMP challenger and remote 30-minute monitor loop.
- Phenomenon: Monitor session remains alive; training PID 2566681 remains on GPU1 and has advanced to iteration 19241.
- Analysis: Checkpoint_19200 exists (48 MB). Latest row has vx/yaw MAE `0.159/0.226`, vx/yaw correlation `0.949/0.835`, value loss `0.102`, PPO update fraction `0.539`, termination `0`, and AMP score `-0.287`; operational health is good but style reward is still negative.
- Adjustment: None; keep monitoring at 1800-second cadence and avoid concurrent Isaac playback.
- Rationale: Preserve one challenger and collect a comparable post-19200 window before deciding whether command-matched AMP improves expert gait.
- Expected effect: Continued directional response learning with no GPU interference from evaluation.
- Result: Monitor loop corrected and persistent; `checkpoint_19200.pt` is the latest verified checkpoint.
- Files/commands: monitor `/tmp/bfm_ampcmdmatched_monitor_30m.log`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_ampcmdmatched_resume19100_20260902`; tmux `bfm_ampcmdmatched_resume19100_20260902`.

## 2026-09-02 15:45:49 CST - Reject command-matched AMP and roll back to 19100 baseline

- Context: PiPlus Stage2 weighted14, GPU11 physical GPU1, checkpoint_19100, run amp_baseline_rollback19100_20260902
- Phenomenon: At checkpoint_19500 the command-matched challenger had 100-row means vx/vy/yaw MAE 0.139/0.106/0.211, command correlations 0.874/0.810, AMP score -0.243, discriminator gap 0.779; baseline 19100 window had 0.127/0.102/0.229, correlations 0.885/0.778, AMP -0.154, gap 0.059.
- Analysis: Command-category expert sampling improved yaw correlation but drove discriminator saturation and worsened AMP style and forward/lateral errors; MuJoCo fixed-command playback also showed severe simulator mismatch, so it is not a valid Isaac acceptance gate.
- Adjustment: Stopped bfm_ampcmdmatched_resume19100_20260902; restored no-command-matched baseline semantics and resumed exact checkpoint_19100 with weighted14, smooth005, policy LR 2e-6, discriminator LR 1e-5, AMP 0.03.
- Rationale: Keep the last comparable champion alive and remove the evidenced adversarial distribution shift before testing another hypothesis.
- Expected effect: Return discriminator gap toward zero while preserving low termination and command tracking; monitor checkpoint-aligned AMP, per-axis MAE/correlation, PPO KL/update fraction, and stand contribution.
- Result: Rollback launched and independently verified on GPU11 physical GPU1 as PID 2577005; 30-minute monitor tmux bfm_amp_baseline_monitor_30m is active.
- Files/commands: Remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_rollback19100_20260902; monitor /tmp/bfm_amp_baseline_monitor_30m.log; evaluator logs /tmp/eval19500/

## 2026-09-02 15:56:49 CST - Add lateral velocity response observability and resume baseline

- Context: PiPlus Stage2 weighted14 baseline, checkpoint_19100, GPU11 physical GPU1, run baseline_metrics_resume19100_20260902
- Phenomenon: Existing tracking logs exposed vx/yaw response but no vy slope, correlation, or left/right slices, preventing evidence-based lateral tuning.
- Analysis: User target includes all-direction velocity response. Missing lateral metrics is an observability gap; adding diagnostics does not alter policy inputs, rewards, checkpoint tensors, or action semantics.
- Adjustment: Added tracking/nonzero_vy_response_slope, tracking/nonzero_vy_command_correlation, tracking/nonzero_achieved_vy_std, tracking/vy_bin_left_mae, and tracking/vy_bin_right_mae; added regression test; remote backup /tmp/ht_bfm_backup_tracking_metrics_20260902.py; restarted from exact checkpoint_19100 with baseline no-command-matched semantics.
- Rationale: Expose direct lateral response and sign asymmetry before selecting a future command-distribution or reward challenger.
- Expected effect: Each checkpoint window reports vx/vy/yaw response separately; no performance regression because training semantics are unchanged.
- Result: Remote py_compile and 29 unit tests pass. Fresh process PID 2579602 is alive on CUDA_VISIBLE_DEVICES=1; first rows show vy correlation 0.84-0.87, slope 0.65-0.75, termination 0.
- Files/commands: humanoidverse/amp_stage2.py; tests/test_amp_stage2.py; remote tmux bfm_amp_baseline_metrics_resume19100_20260902; monitor /tmp/bfm_amp_metrics_monitor_30m.log

## 2026-09-02 16:31:54 CST - Launch lateral-command sampling challenger from 19300

- Context: PiPlus Stage2 weighted14, checkpoint_19300, GPU11 physical GPU1, run lateral15_resume19300_20260902
- Phenomenon: Baseline near-window lateral metrics showed persistent right-side error (vy bin right MAE ~0.158 vs left ~0.117) and no explicit lateral-only command coverage.
- Analysis: The command sampler was uniform planar plus stand/turn gates; a conservative 0.15 lateral-only probability should increase +/-vy learning while preserving stand and turn probabilities. This is a single command-distribution mutation; reward, optimizer, policy architecture, and expert data are unchanged.
- Adjustment: Added optional --command-lateral-prob (default 0); lateral-only samples zero vx/wz and bypass stand gate; launched with 0.15 from complete checkpoint_19300.
- Rationale: Increase direct lateral response samples to test whether the observed right-side MAE asymmetry improves without collapsing forward/yaw tracking or AMP balance.
- Expected effect: At checkpoint-aligned maturity, lower vy and right-bin MAE with stable vx/yaw MAE, correlations, AMP gap, PPO health, and zero terminations. Reject if tracking/style/safety regresses.
- Result: Early iteration 19303: vy MAE 0.090, right-bin 0.151, vy correlation 0.874, termination 0; challenger pending mature checkpoint.
- Files/commands: humanoidverse/amp_stage2.py; tests/test_amp_stage2.py; remote backups /tmp/ht_bfm_backup_lateral_sampling_20260902.py(.test.py); tmux bfm_amp_lateral15_resume19300_20260902; monitor /tmp/bfm_amp_lateral15_monitor_30m.log

## 2026-09-02 16:33:58 CST - Repair lateral challenger launch argument

- Context: PiPlus Stage2 lateral15 challenger from checkpoint_19300
- Phenomenon: First detached launch exited because remote source had not yet received the new --command-lateral-prob parser option.
- Analysis: Launch-path race between patch application and tmux start; no training worker advanced and no checkpoint was modified.
- Adjustment: Restored pre-lateral backup, applied the lateral sampler/parser/metadata/call patch atomically, reran py_compile and 29 remote tests, then relaunched the unchanged lateral15 command.
- Rationale: Repair launch mechanism without stacking a second semantic change.
- Expected effect: Stable lateral15 process with lateral-only command probability 0.15 and full metric logging.
- Result: Fresh PID 2588671 alive at iteration 19314; no termination/fall, vy correlation 0.87 and right-bin MAE 0.154 on early rows; mature checkpoint pending.
- Files/commands: Remote backup /tmp/ht_bfm_backup_lateral_sampling_20260902.py; tmux bfm_amp_lateral15_resume19300_20260902

## 2026-09-02 16:35:06 CST - Lateral15 early checkpoint window remains pending

- Context: PiPlus Stage2 lateral15 challenger, checkpoint_19300 resume, GPU11 physical GPU1
- Phenomenon: At iteration 19314 the challenger is alive with no terminations; vy correlation 0.870, slope 0.628, left/right MAE 0.110/0.154, vx/yaw MAE 0.113/0.197.
- Analysis: Right-side lateral error is slightly below the pre-change 19300 baseline snapshot, but the window is only 14 updates and AMP gap 0.393 is not mature evidence.
- Adjustment: None; keep the single challenger running under the 1800-second monitor loop.
- Rationale: Avoid promoting or stacking changes before checkpoint-aligned 19400+ evidence.
- Expected effect: Assess mature lateral response, AMP style, PPO health, and safety against the 19300 baseline using the new vy metrics.
- Result: Pending checkpoint-aligned evaluation.
- Files/commands: remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_lateral15_resume19300_20260902/console.log; monitor /tmp/bfm_amp_lateral15_monitor_30m.log

## 2026-09-02 16:38:28 CST - Lateral15 challenger early health check

- Context: PiPlus Stage2 lateral15_resume19300_20260902, GPU11 physical GPU1
- Phenomenon: At iteration 19334 the lateral sampler challenger remains alive without fall/termination; vx/vy/yaw MAE 0.115/0.093/0.202, vy correlation 0.860 and slope 0.656, left/right vy MAE 0.110/0.151, AMP score -0.238, discriminator gap 0.384.
- Analysis: Lateral response is directionally similar to or slightly better than the baseline early window, while AMP gap is moderate and PPO update fraction reached 1.0 on the latest row. This is not yet a checkpoint-aligned maturity result.
- Adjustment: None; keep the single challenger and 1800-second monitor loop active.
- Rationale: Avoid premature promotion or a second mutation before checkpoint_19400+ and a repeat fixed playback protocol.
- Expected effect: Confirm sustained right-side improvement without forward/yaw, AMP, or safety regression; then test an Isaac-native stand velocity diagnostic.
- Result: Pending mature checkpoint.
- Files/commands: remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_lateral15_resume19300_20260902/console.log; /tmp/bfm_amp_lateral15_monitor_30m.log

## 2026-09-02 16:51:28 CST - Reject lateral15 and recover from stable 19200 checkpoint

- Context: PiPlus Stage2 weighted14 baseline; lateral15 challenger from 19300; GPU11 physical GPU1
- Phenomenon: Lateral15 reached 19391 with yaw MAE 0.259, yaw correlation 0.642, vy slope 0.484, value loss 0.94. Resuming baseline from 19300 reproduced unstable early windows (19319-19331), while checkpoint_19200 recovery at 19201-19204 returned to vx/vy/yaw MAE 0.108/0.081/0.202 with zero termination.
- Analysis: The 19300 state was not a reliable rollback point; 19200 is the latest checkpoint with an independently observed stable continuation. Lateral-only command oversampling is rejected as a performance challenger.
- Adjustment: Stopped lateral15 and baseline-after-lateral runs; resumed unchanged baseline semantics from checkpoint_19200 in baseline_recovery19200_20260902. Retained only optional lateral metrics instrumentation, with lateral sampler default 0.
- Rationale: Restore a stable champion before any stand-still or expert-style intervention.
- Expected effect: Continue healthy training from 19200; use checkpoint-aligned windows to diagnose stand velocity/oscillation before changing rewards.
- Result: PID 2593769 alive at iteration 19204, CUDA_VISIBLE_DEVICES=1, termination/fall 0; 30-minute monitor bfm_amp_baseline_recovery_monitor_30m active.
- Files/commands: Remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_recovery19200_20260902; monitor /tmp/bfm_amp_baseline_recovery_monitor_30m.log

## 2026-09-02 17:14:27 CST - Stand velocity reward challenger checkpoint 19300 screen

- Context: PiPlus Stage2 standvel_reward_resume19200_20260902, checkpoint_19300, GPU11 physical GPU1
- Phenomenon: The stand-velocity reward challenger reached checkpoint_19300 with zero terminations. Latest near-window stand planar speed RMS 0.0319 and yaw-rate abs mean 0.1115; vx/vy/yaw MAE 0.112/0.090/0.194. Baseline 19200-19204 showed stand RMS 0.042-0.125 and yaw abs mean 0.124-0.166 during warmup.
- Analysis: The new terms directly penalize zero-command root drift using normalized/clipped planar and yaw rates; early Isaac evidence is compatible with lower stand drift, while task tracking remains within baseline variation. MuJoCo fixed-command playback remains a diagnostic only because of known simulator mismatch.
- Adjustment: Added MimicLite terms stand_linvel_l2 and stand_yawvel_l2, each weight 0.5, gated to standing commands; retained lateral/vy and stand metrics; resumed from checkpoint_19200.
- Rationale: Give the policy an explicit training gradient against root drift, addressing the static-standing requirement without action routing or checkpoint-shape changes.
- Expected effect: At mature checkpoints, reduce stand_base_planar_speed_rms and stand_base_yaw_rate_abs_mean while preserving vx/vy/yaw tracking, AMP gap, PPO health, and zero falls.
- Result: Checkpoint_19300 is valid and short five-command playback completed; challenger remains pending maturity. Process PID 2596832 alive; monitor tmux bfm_amp_standvel_reward_monitor_30m active.
- Files/commands: humanoidverse/amp_stage2.py; tests/test_amp_stage2.py; remote backup /tmp/ht_bfm_backup_stand_vel_reward_20260902.py; evaluator /tmp/eval_standvel19300/

## 2026-09-02 17:27:26 CST - Reject stand velocity reward and restore 19100 baseline

- Context: PiPlus Stage2 standvel_reward_resume19200_20260902; GPU11 physical GPU1
- Phenomenon: Stand reward challenger reached checkpoint_19400 but 19358-19457 window regressed: stand RMS last 0.158, vx/vy/yaw MAE last 0.165/0.126/0.236, AMP gap 0.637. Early lower stand drift was not sustained.
- Analysis: The added root-velocity penalties had small contribution scale and did not reliably suppress oscillation; continued training violated task-tracking and stand guardrail trends. MuJoCo fixed-command playback remains non-authoritative due known simulator mismatch.
- Adjustment: Stopped standvel_reward challenger; removed stand_linvel_l2 and stand_yawvel_l2 reward terms locally/remotely; retained stand root-speed observability metrics; resumed unchanged weighted14 baseline from validated checkpoint_19100.
- Rationale: Keep only evidence-backed observability and return to the last comparable stable champion before testing another architecture-level hypothesis.
- Expected effect: Stable vx/vy/yaw response and AMP behavior while stand metrics continue to expose root drift for future targeted changes.
- Result: Remote baseline PID 2603923 alive under tmux bfm_amp_baseline_final19100_20260902; 30-minute monitor bfm_amp_baseline_final_monitor_30m active. Remote 28 tests pass after baseline restore.
- Files/commands: Remote backup /tmp/ht_bfm_backup_stand_vel_reward_20260902.py; current baseline logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_final19100_20260902

## 2026-09-02 17:29:58 CST - Restore 19100 baseline after stand-reward rejection

- Context: PiPlus Stage2 weighted14 baseline_final19100_20260902, GPU11 physical GPU1
- Phenomenon: Stand-velocity reward challenger did not sustain lower root drift through the mature window and worsened tracking at later iterations; reward contributions were small and oscillation persisted.
- Analysis: The last comparable stable champion is the no-stand-reward weighted14 checkpoint_19100. Stand root-speed metrics remain as evaluation-only observability, while reward semantics and checkpoint contract are restored.
- Adjustment: Removed stand_linvel_l2 and stand_yawvel_l2 reward terms locally/remotely; resumed checkpoint_19100 with weighted14, command smoothing 0.05, policy LR 2e-6, discriminator LR 1e-5, AMP 0.03, stand/turn 0.20/0.20.
- Rationale: Avoid repeatedly training unstable challengers while preserving direct evidence needed for the next architecture-level standing solution.
- Expected effect: Stable command tracking and AMP discriminator behavior; collect stand speed metrics and identify whether oscillation originates from frozen BFM latent/action mapping rather than reward shaping.
- Result: Fresh PID 2603923 alive after independent reconnect at iteration 19125; termination/fall zero. 30-minute monitor bfm_amp_baseline_final_monitor_30m active.
- Files/commands: Remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_final19100_20260902; monitor /tmp/bfm_amp_baseline_final_monitor_30m.log

## 2026-09-02 17:32:40 CST - Baseline 19100 post-rollback health check

- Context: PiPlus Stage2 baseline_final19100_20260902, GPU11 physical GPU1
- Phenomenon: After restoring no-stand-reward semantics, iteration 19155 is alive with zero termination/fall; vx/vy/yaw MAE 0.112/0.089/0.198, correlations 0.952/0.847, stand planar RMS 0.046, stand yaw abs mean 0.127, value loss 0.075.
- Analysis: The baseline recovers stable command tracking and low stand drift compared with the rejected stand reward late window. The remaining stand yaw motion is measurable but not yet solved; further intervention requires a validated latent/action-level hypothesis.
- Adjustment: None; keep baseline as active champion and continue 30-minute monitoring.
- Rationale: Do not stack another mutation while the restored champion is healthy and the evaluator has known MuJoCo/Isaac mismatch.
- Expected effect: Accumulate checkpoint-aligned baseline evidence before testing expert-style/stand latent constraints.
- Result: PID 2603923 survives independent reconnect; monitor tmux bfm_amp_baseline_final_monitor_30m active.
- Files/commands: Remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_final19100_20260902/console.log; /tmp/bfm_amp_baseline_final_monitor_30m.log

## 2026-09-02 17:33:31 CST - Post-rollback baseline health and stand metric retention

- Context: PiPlus Stage2 baseline_final19100_20260902, GPU11 physical GPU1
- Phenomenon: Fresh baseline from checkpoint_19100 reached iteration 19155 with zero termination/fall, vx/vy/yaw MAE 0.112/0.089/0.198, stand planar RMS 0.046 and stand yaw abs mean 0.127; no reward mutation active.
- Analysis: Restored baseline gives the cleanest comparable champion after rejecting command-matched AMP, lateral oversampling, and stand velocity reward challengers. Stand velocity observability remains available to identify whether future changes improve real Isaac behavior.
- Adjustment: None; continue baseline training and 30-minute supervision.
- Rationale: Preserve a stable reference while designing a latent/action-level expert-style standing hypothesis.
- Expected effect: Maintain operational health and checkpoint progression without masking failures under aggregate reward.
- Result: PID 2603923 alive after independent reconnect; tmux bfm_amp_baseline_final_monitor_30m remains active.
- Files/commands: Remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_final19100_20260902/console.log; monitor /tmp/bfm_amp_baseline_final_monitor_30m.log

## 2026-09-02 18:59:09 CST - Reject broad and strict stand-latent anchors; recover baseline 19200

- Context: PiPlus Stage2 weighted14, checkpoint_19200, GPU11 physical GPU1
- Phenomenon: Broad stand latent anchor (coef 0.002, <0.1 command mask) collapsed by iteration 19293: vx/yaw corr 0.08/-0.10, stand RMS 0.305, value loss 1.72. Strict-zero anchor (coef 0.001, <0.02 mask) was initially stable but collapsed by 19366: vx/yaw corr 0.53/0.64, value loss 1.02.
- Analysis: The BFM backward-map default latent is measurably different from the policy zero-command latent (cosine 0.011, action difference 0.103), but direct latent anchoring conflicts with PPO/AMP dynamics even after excluding near-zero moving commands. Treat this as evidence against latent-only regularization, not a reason for more coefficient search.
- Adjustment: Stopped both anchor challengers, removed stand_latent auxiliary loss and related CLI/rollout state, retained only stand/lateral observability metrics, and resumed unchanged baseline from checkpoint_19200.
- Rationale: Restore the stable champion and avoid repeating a failed latent-only architecture family; future stand work must use validated action-level or reset-state supervision with a separate evaluator.
- Expected effect: Stable baseline tracking and AMP behavior with direct stand root-speed metrics available.
- Result: PID 2622600 alive on CUDA_VISIBLE_DEVICES=1; baseline recovery monitor tmux bfm_amp_baseline_recover19200_monitor_30m active. /tmp/IsaacLab stale cache cleanup freed 20GB after launch blocker.
- Files/commands: Remote backups /tmp/ht_bfm_backup_anchor_20260902.py(.test.py); current run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_recover19200_20260902; monitor /tmp/bfm_amp_baseline_recover19200_monitor_30m.log

## 2026-09-02 19:08:56 CST - Repair anchor launch blocker and recover single baseline lineage

- Context: GPU11 JumpServer, HT_BFM Stage2, checkpoint_19200
- Phenomenon: Anchor challenger first failed in Isaac contact-sensor init; retry failed because /tmp/IsaacLab consumed 20GB and overlay reached 100%.
- Analysis: Both failures were launch/resource defects, not model evidence. No training iterations were produced by the failed launch. Stale /tmp/IsaacLab USD directories had no open file descriptors from active processes.
- Adjustment: Removed 233 stale generated /tmp/IsaacLab directories older than 30 minutes (freed ~20GB), stopped anchor lineage, restored baseline source, and resumed baseline from checkpoint_19200.
- Rationale: Keep remote training operational and preserve checkpoint lineage while avoiding unrelated GPU/process changes.
- Expected effect: Baseline progresses on GPU1 with 30-minute supervision; future mutations start only after clean resource verification.
- Result: Baseline recovery PID 2622600 remains alive at iteration 19228; overlay free space ~19GB; monitor bfm_amp_baseline_recover19200_monitor_30m active.
- Files/commands: Remote logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_recovery19200_20260902; /tmp/bfm_amp_baseline_recover19200_monitor_30m.log

## 2026-09-02 19:13:47 CST - Re-establish baseline after anchor rejection

- Context: PiPlus Stage2 weighted14 baseline_final_recover19200_20260902, checkpoint_19200, GPU11 physical GPU1
- Phenomenon: Both latent-anchor variants were rejected for delayed tracking collapse despite temporarily lower stand drift. The no-anchor source was restored and the previous baseline lineage was stopped.
- Analysis: Current source contains only command-category/lateral and stand root-speed observability; no new reward, latent constraint, or action routing. This is recovery-only, not a performance claim.
- Adjustment: Started baseline_final_recover19200_20260902 from complete checkpoint_19200 with original weighted14 settings and 2e-6/1e-5 optimizers.
- Rationale: Maintain a validated champion while preserving evidence for a future action-level standing fix.
- Expected effect: Stable training and clean three-axis/stand metrics under the original Stage2 semantics.
- Result: Independent reconnect confirms PID 2625502 on CUDA_VISIBLE_DEVICES=1; latent contract check passed. Monitor bfm_amp_baseline_final_recover_monitor_30m active.
- Files/commands: Remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_final_recover19200_20260902/console.log; monitor /tmp/bfm_amp_baseline_final_recover_monitor_30m.log

## 2026-09-02 19:27:49 CST - Correct resume path and re-establish 19100 champion

- Context: PiPlus Stage2 weighted14, GPU11 physical GPU1
- Phenomenon: A recovery launch pointed to a nonexistent checkpoint_19100 inside baseline_final19100; authoritative checkpoint_19100 exists in weighted14_baseline_resume19000_20260902. The process exited before training.
- Analysis: This was a launch-path defect, not a model failure. Using the verified checkpoint location restores the prior stable baseline lineage.
- Adjustment: Stopped stale 19200 recovery process, launched baseline_champion19100_retry_20260902 from logs/amp_stage2_piplus_lse_1gpu_4096env_1m_weighted14_baseline_resume19000_20260902/checkpoint_19100.pt with no-anchor semantics and current stand/lateral observability.
- Rationale: Prevent accidental resume from an incomplete or wrong directory while maintaining a single champion.
- Expected effect: Clean progression from checkpoint_19100 with zero termination and comparable tracking/AMP windows.
- Result: PID 2628738 alive after independent reconnect; initial iteration 19102 metrics match prior stable baseline (vx/yaw corr 0.959/0.854, stand RMS 0.067, termination 0). Monitor bfm_amp_baseline_champion19100_retry_monitor_30m active.
- Files/commands: Remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_champion19100_retry_20260902; monitor /tmp/bfm_amp_baseline_champion19100_retry_monitor_30m.log

## 2026-09-02 19:29:32 CST - 19100 champion retry early warmup

- Context: PiPlus Stage2 baseline_champion19100_retry_20260902, GPU11 physical GPU1
- Phenomenon: After correct checkpoint path, 20 updates reached iteration 19118 with termination 0, value loss 0.094, discriminator gap 0.114; latest vx/vy/yaw MAE 0.152/0.121/0.245 and stand RMS 0.158 during warmup.
- Analysis: Early warmup is noisier than the prior 19102 row but remains numerically finite and has no falls. Do not judge promotion/rejection from this short window.
- Adjustment: None; keep one champion lineage and 30-minute monitor active.
- Rationale: Accumulate a mature checkpoint-aligned window before any new expert-style or standing change.
- Expected effect: Baseline settles toward the previously observed 19100-19155 behavior while preserving direct stand/lateral metrics.
- Result: PID 2628738 alive after independent reconnect at iteration 19118.
- Files/commands: Remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_champion19100_retry_20260902/console.log; monitor /tmp/bfm_amp_baseline_champion19100_retry_monitor_30m.log

## 2026-09-02 19:30:36 CST - 19100 champion retry confirmed and monitor upgraded

- Context: PiPlus Stage2 weighted14, GPU11 physical GPU1, baseline_champion19100_retry_20260902
- Phenomenon: Corrected launch now progresses through iteration 19129; early 19102 metrics matched prior stable baseline. Previous 19200 recovery showed delayed PPO/yaw oscillation.
- Analysis: Checkpoint_19100 has the cleanest verified optimizer state. Monitoring now extracts the latest JSON metric row instead of raw log tail, so Isaac warnings cannot hide iteration/health.
- Adjustment: No training semantic change; maintained weighted14 baseline, 2e-6 policy LR, 1e-5 discriminator LR, AMP 0.03, smoothing 0.05. Monitor loop uses grep ^{ | tail -1.
- Rationale: Preserve a valid champion and make the requested half-hour supervision actionable.
- Expected effect: Stable progression with comparable vx/vy/yaw, AMP gap, value loss, termination, and stand root-speed metrics.
- Result: PID 2628738 alive on CUDA_VISIBLE_DEVICES=1 at iteration 19129; monitor bfm_amp_baseline_champion19100_retry_monitor_30m active.
- Files/commands: Remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_champion19100_retry_20260902/console.log; /tmp/bfm_amp_baseline_champion19100_retry_monitor_30m.log

## 2026-09-02 19:31:43 CST - 19100 champion retry settles after warmup

- Context: PiPlus Stage2 baseline_champion19100_retry_20260902, GPU11 physical GPU1
- Phenomenon: At iterations 19136-19140 the corrected 19100 resume remains alive with zero terminations; latest 19140 has vx/vy/yaw MAE 0.108/0.084/0.197, vx/yaw correlations 0.911/0.796, stand planar RMS 0.121 and stand yaw abs mean 0.163, value loss 0.096.
- Analysis: The initial 19118 warmup spike settled back to the prior stable range. AMP score remains negative (~-0.21) and discriminator gap ~0.12, so expert-style improvement is not yet demonstrated.
- Adjustment: None; keep this single baseline champion and 30-minute monitor active.
- Rationale: Preserve a reproducible baseline while avoiding further rejected latent/reward mutations.
- Expected effect: Accumulate a mature checkpoint window and use direct stand/lateral metrics to guide any next action-level change.
- Result: Pending mature checkpoint and fixed evaluation.
- Files/commands: Remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_champion19100_retry_20260902/console.log; monitor /tmp/bfm_amp_baseline_champion19100_retry_monitor_30m.log

## 2026-09-02 19:44:17 CST - 19100 champion reaches checkpoint 19200

- Context: PiPlus Stage2 weighted14 baseline_champion19100_retry_20260902, GPU11 physical GPU1
- Phenomenon: Corrected 19100 resume reached checkpoint_19200 and iteration 19204 without terminations. The 19116-19195 window averaged vx/vy/yaw MAE 0.128/0.103/0.227, vx/yaw correlation 0.886/0.778, stand RMS 0.099, stand yaw 0.163, value loss 0.224, AMP score -0.230, discriminator gap 0.147.
- Analysis: The 19100 lineage is operationally reproducible and has better latest-row recovery than the 19200 recovery lineage. AMP style remains below expert (negative score), and stand yaw drift remains measurable; no safe promotion to final behavior is proven.
- Adjustment: None; preserve checkpoint_19200 as the current compatible rollback while continuing from 19100 lineage under the original semantics.
- Rationale: Avoid new unvalidated mutations after two latent-anchor failures; collect checkpoint-aligned baseline evidence and design a new action-level style hypothesis.
- Expected effect: Stable ongoing training with direct stand/lateral/three-axis metrics and 30-minute monitor samples.
- Result: PID 2628738 alive at iteration 19204; checkpoint_19200 complete; monitor bfm_amp_baseline_champion19100_retry_monitor_30m active.
- Files/commands: Remote checkpoint logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_champion19100_retry_20260902/checkpoint_19200.pt; /tmp/bfm_amp_baseline_champion19100_retry_monitor_30m.log

## 2026-09-02 19:47:47 CST - Baseline 19100 champion checkpoint 19200 active

- Context: PiPlus Stage2 weighted14 baseline_champion19100_retry_20260902, GPU11 physical GPU1
- Phenomenon: Current 100-row window (19135-19234) shows vx/vy/yaw MAE 0.127/0.102/0.224, vx/yaw correlation 0.896/0.788, stand RMS 0.094, stand yaw abs mean 0.156, value loss 0.207, termination 0.
- Analysis: The corrected 19100 checkpoint lineage is operationally healthy and materially more stable than 19200-resume and latent-anchor challengers, but AMP score remains negative and direct expert-style/zero-command immobility is not yet proven.
- Adjustment: None; keep checkpoint_19200 as compatible rollback and continue the single champion under the upgraded JSON monitor.
- Rationale: Avoid stacking another mutation until a mature baseline behavior window is available for a new architecture hypothesis.
- Expected effect: Stable training and reliable half-hour process/checkpoint/metric supervision.
- Result: PID 2628738 alive at iteration 19234; checkpoint_19200 complete; monitor bfm_amp_baseline_champion19100_retry_monitor_30m active.
- Files/commands: Remote console logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_champion19100_retry_20260902/console.log; /tmp/bfm_amp_baseline_champion19100_retry_monitor_30m.log

## 2026-09-02 20:13:06 CST - Conditional AMP challenger checkpoint 19200 screen

- Context: PiPlus Stage2 ampconditional002_resume19100_20260902, checkpoint_19200, GPU11 physical GPU1
- Phenomenon: Conditional discriminator reached checkpoint_19200. 19108-19207 window: AMP score -0.080 and gap 0.031 versus baseline -0.250/0.187; vx/vy/yaw MAE 0.127/0.103/0.225 versus baseline 0.127/0.102/0.224; termination 0. Stand RMS 0.103 versus baseline 0.094.
- Analysis: Command conditioning materially improves direct AMP/discriminator signals without aggregate tracking regression. Stand drift is slightly worse, and WGAN gradient norm is still warming from ~0.1 toward 1, so promotion is premature.
- Adjustment: Keep conditional AMP challenger running with fresh 205-D discriminator, AMP weight 0.02, policy/discriminator LR 2e-6/1e-5; no further mutation.
- Rationale: Allow the fresh conditional discriminator to mature while preserving task/safety guardrails.
- Expected effect: At checkpoint_19300+, retain lower discriminator gap and improve expert style without stand or tracking regression.
- Result: Checkpoint_19200 valid; five-command MuJoCo screen completed with known simulator mismatch and one backward termination; Isaac training remains termination-free. Challenger pending.
- Files/commands: Remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_ampconditional002_resume19100_20260902; /tmp/eval_ampconditional19200/; monitor /tmp/bfm_amp_ampconditional002_monitor_30m.log

## 2026-09-02 20:18:59 CST - Reject conditional AMP for delayed tracking regression and restore baseline

- Context: PiPlus Stage2 weighted14, command-conditioned discriminator challenger from checkpoint_19100, GPU11 physical GPU1
- Phenomenon: Conditional AMP improved 100-row AMP score/gap (-0.080/0.031 vs baseline -0.250/0.187), but by 19260 vx/vy/yaw MAE reached 0.161/0.163/0.240 and stand RMS 0.212; direct response degraded despite zero termination.
- Analysis: Command conditioning fixed the unconditional discriminator mismatch but fresh discriminator dynamics shifted the policy toward style at the expense of direct command tracking and stand stability. This fails the task guardrail; do not promote based on AMP alone.
- Adjustment: Stopped ampconditional002; restored no-conditional source from remote backup and resumed baseline from checkpoint_19100 at baseline_after_conditional_reject19100_20260902.
- Rationale: Preserve direct task behavior and safety while retaining conditional implementation as a reversible experiment for future redesign.
- Expected effect: Baseline recovers vx/vy/yaw tracking and lower stand drift; future style work needs a multi-objective/dual-discriminator or frozen-style diagnostic approach.
- Result: PID 2641826 alive at iteration 19112, termination/fall 0; 30-minute monitor bfm_amp_baseline_after_conditional_monitor_30m active; 29 remote tests passed after source rollback.
- Files/commands: Conditional run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_ampconditional002_resume19100_20260902; baseline logs/amp_stage2_piplus_lse_1gpu_4096env_1m_baseline_after_conditional_reject19100_20260902; backup /tmp/ht_bfm_backup_conditional_amp_20260902.py

## 2026-09-02 20:21:56 CST - Conditional AMP rejected on direct behavior guardrail

- Context: PiPlus Stage2 ampconditional002_resume19100_20260902, GPU11 physical GPU1
- Phenomenon: Command-conditioned discriminator improved AMP score/gap but delayed window (19161-19260) worsened vx/vy/yaw response and stand drift; latest MAE 0.161/0.163/0.240, stand RMS 0.212.
- Analysis: Style-only discriminator signal is not sufficient; direct command tracking and static stability remain primary task guardrails. Fresh discriminator dynamics caused policy regression despite near-zero conditional gap.
- Adjustment: Stopped conditional challenger, restored baseline source from /tmp/ht_bfm_backup_conditional_amp_20260902.py, resumed checkpoint_19100 at baseline_after_conditional_reject19100_20260902.
- Rationale: Reject a model that looks better only on AMP metrics and preserve the stable task-behavior champion.
- Expected effect: Recover original three-axis tracking and stand metrics while retaining conditional implementation only as a documented reversible experiment.
- Result: PID 2641826 alive at iteration 19112; termination/fall 0; monitor bfm_amp_baseline_after_conditional_monitor_30m active; 29 remote tests pass.
- Files/commands: Conditional logs ampconditional002_resume19100_20260902; baseline logs baseline_after_conditional_reject19100_20260902; monitor /tmp/bfm_amp_baseline_after_conditional_monitor_30m.log

## 2026-09-02 21:20 CST - Accept style02 discriminator-reset challenger at checkpoint 19500

- Context: PiPlus Stage2 weighted14, GPU11 physical GPU1; baseline checkpoint_19400 versus style02 checkpoint_19500.
- Baseline evidence: last-100 training rows had vx/vy/yaw MAE 0.184/0.138/0.233, stand planar RMS/yaw 0.087/0.141, AMP score -0.279, discriminator gap 0.549, PPO update fraction 0.710, termination 0.
- Hypothesis: stale discriminator state was saturating against the weighted expert distribution; a fresh discriminator plus smaller adversarial/update steps can improve expert style without moving the command encoder off the stable basin.
- Adjustment: derived `checkpoint_19400_discreset_style02.pt` from the complete baseline checkpoint, replacing only the discriminator and removing its optimizer state; launched with AMP weight 0.03 -> 0.02, discriminator LR 1e-5 -> 2e-6, policy LR 2e-6 -> 1e-6, PPO epochs 3 -> 2. Action/observation/robot/expert dataset contracts unchanged.
- Challenger window 19400-19500: vx/vy/yaw MAE 0.184/0.136/0.229, stand planar RMS/yaw 0.084/0.137, AMP score +0.031, discriminator gap 0.008, PPO update fraction 0.980, termination 0.
- Fixed Isaac headless evaluation: checkpoint 19500, seed 1, 200 steps for stand/forward/backward/lateral/yaw and 500-step long screens for lateral/yaw; no terminations. Stand settled to <=0.004 m/s planar and <=0.001 rad/s yaw; forward/backward reached +/-0.24-0.38 m/s; lateral transiently overshot 0.40 then settled 0.25; yaw reached 0.54-0.71 rad/s for 0.6 target. Initial Isaac attempts failed with OpenBLAS fork segfault; rerun with OPENBLAS/OMP/MKL/NUMEXPR=1 succeeded.
- Decision: accept checkpoint_19500 as champion and resume the same configuration; no local source change. Baseline checkpoint_19400 backup: `/tmp/ht_bfm_backup_style02_20260902/checkpoint_19400.pt`; derived checkpoint SHA256 `b326bb5560eec0851a2b7b84bf444015a6e8d31364a80fe2ba01903340a15ede`.
- Evaluation artifacts: remote `/tmp/eval_isaac19400_retry/` and `/tmp/eval_isaac19500/`; baseline checkpoint SHA256 `124ce6d80ea42834e8dd65475a182827fec352263eeadde5261e7e6a7502c456`.
- Result: accepted; resumed champion verification pending fresh SSH reconnect. Keep 19500 as rollback point and monitor next 100-row window for yaw lateral cross-axis drift and stand RMS.

- Post-restart proof: launch SSH was closed and a new JumpServer/GPU11 connection established. PID 2660134 remains alive on physical GPU1 (`CUDA_VISIBLE_DEVICES=1`); at reconnect it advanced from iteration 19500 to 19517 with PPO update fraction 1.0, value loss 0.10, termination rate <1e-5, vx/vy/yaw MAE 0.169/0.123/0.215, stand RMS/yaw 0.058/0.119, AMP score 0.005, discriminator gap 0.028. GPU0 and GPU2 process mappings are unchanged and excluded.

## 2026-09-02 22:05 CST - 19600 champion fixed Isaac screen and continuation

- Context: style02 discriminator-reset champion, checkpoint_19600, GPU11 physical GPU1; no new training mutation.
- Fixed Isaac protocol: `humanoidverse.amp_stage2_play`, `--simulator isaacsim --device cuda:0 --policy-device cpu --headless --no-realtime --max-steps 300 --log-every-steps 100`, seed 1, commands stand `(0,0,0)`, forward/backward `(+-0.4,0,0)`, lateral `(0,0.25,0)`, yaw `(0,0,0.6)`, thread isolation OPENBLAS/OMP/MKL/NUMEXPR=1, CUDA_VISIBLE_DEVICES=1. Logs `/tmp/eval_isaac19600/`.
- Play metrics: all five scenarios completed with no termination. Stand settled to approximately `0.002 m/s` planar and `0.04 rad/s` yaw by step 300. Forward/backward reached `+0.35/-0.34 m/s`; lateral `0.32-0.37 m/s` (transient over target, continued prior pattern); yaw `0.50-0.70 rad/s` for 0.6 target. Training window 19500-19600 remained finite with termination ~0, PPO update fraction 0.985, vx/vy/yaw MAE 0.178/0.132/0.231, stand RMS/yaw 0.091/0.146, AMP score -0.005, discriminator gap 0.034.
- Analysis: Static standing and survival remain improved; yaw response is materially closer to target than the old baseline. Lateral transient overshoot is the remaining guardrail to monitor, but it settles and has not produced falls. No new scalar change is justified until a longer window distinguishes a persistent lateral bias from command-transition transient.
- Action: Stopped only the selected process after checkpoint_19600, then resumed unchanged champion parameters (`amp_weight=0.02`, discriminator LR `2e-6`, policy LR `1e-6`, PPO epochs `2`, smoothing `0.05`) from the verified complete checkpoint. New run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_style02_resume19600_20260902`, tmux `bfm_style02_resume19600_20260902`.
- Post-disconnect proof: closed launch SSH and reconnected independently; PID 2665388 remained alive on physical GPU1 with the same command and advanced `19600 -> 19606`; PPO update fraction 1.0, value loss 0.097, termination 0, vx/vy/yaw MAE 0.173/0.119/0.204, stand RMS/yaw 0.059/0.115, AMP score -0.026, discriminator gap 0.049. GPU0/2 mappings remained unchanged.

## 2026-09-03 10:35 CST - 19800 fixed Isaac screen and champion resume

- Context: style02 champion, checkpoint_19800 from `style02_resume19700_20260903`, GPU11 physical GPU1.
- Training window 19700-19800: vx/vy/yaw MAE 0.1889/0.1277/0.2330, command correlations 0.871/0.818, stand planar RMS/yaw 0.1157/0.1562, value loss 0.219, PPO update fraction 0.984, termination 1e-6. This is a finite, numerically healthy window but static training metrics are noisier than the fixed Isaac result.
- Fixed Isaac protocol: `humanoidverse.amp_stage2_play --simulator isaacsim --device cuda:0 --policy-device cpu --headless --no-realtime --max-steps 300 --log-every-steps 100`, seed 1, stand/forward/backward/lateral/yaw commands `(0,0,0)/(+0.4,0,0)/(-0.4,0,0)/(0,+0.25,0)/(0,0,+0.6)`, thread isolation OPENBLAS/OMP/MKL/NUMEXPR=1, CUDA_VISIBLE_DEVICES=1; artifacts `/tmp/eval_isaac19800/`.
- Play result: all five scenarios completed without termination. Stand settled to `[-0.005,0.005,-0.064]` velocity by step 300 (about 0.01 m/s planar, 0.064 rad/s yaw). Forward reached 0.419 m/s at step 300, backward -0.290 m/s, lateral 0.359 m/s, yaw 0.490 rad/s; no persistent runaway observed. Relative to 19700, standing remains safe, forward response is slightly stronger, backward response remains weaker, and lateral overshoot persists as the main residual tracking issue.
- Analysis: No new reward/optimizer mutation is justified: lateral-only oversampling already caused delayed yaw collapse in a prior challenger, and standalone stand-velocity rewards caused delayed tracking/AMP regression. Keep the accepted style02 parameters and monitor a longer window; a future lateral fix must address command/representation asymmetry rather than blind scalar weighting.
- Action: Stopped only the selected process after checkpoint_19800, resumed unchanged champion parameters (`amp_weight=0.02`, discriminator LR `2e-6`, policy LR `1e-6`, PPO epochs `2`, smoothing `0.05`) from the complete checkpoint in `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_style02_resume19800_20260903`, tmux `bfm_style02_resume19800_20260903`.
- Post-disconnect proof: launch connection closed and fresh JumpServer/GPU11 reconnect completed; PID 2777867 remains live on physical GPU1 and advanced `19800 -> 19809`; latest metrics vx/vy/yaw MAE 0.178/0.117/0.225, stand RMS/yaw 0.071/0.125, value loss 0.106, PPO update fraction 1.0, termination 0, AMP score -0.097, discriminator gap 0.087. GPU0/2 mappings remained unchanged.

## 2026-09-03 10:55 CST - 19700/19800 evaluation completion and persistent resume

- The previously interrupted 19700 fixed Isaac evaluation was completed: stand settled to about 0.01 m/s planar and 0.009 rad/s yaw; forward/backward reached about +0.30/-0.28 m/s; lateral 0.21-0.32 m/s; yaw 0.52-0.69 rad/s; all scenarios terminated cleanly.
- The 19800 protocol then completed with the same five commands and 300-step horizon. No termination occurred; stand settled to about 0.01 m/s planar and 0.064 rad/s yaw, forward 0.419 m/s, backward -0.290 m/s, lateral 0.359 m/s, yaw 0.490 rad/s. Residual lateral overshoot and backward under-response are documented; no unsupported scalar mutation was added.
- After the 19800 screen, the selected trainer was relaunched unchanged from the complete 19800 checkpoint and the launch SSH was closed. A fresh JumpServer/GPU11 reconnect confirmed PID 2777867 alive in tmux `bfm_style02_resume19800_20260903`, physical GPU1 (`CUDA_VISIBLE_DEVICES=1`), with iteration 19800 -> 19809, termination 0, PPO update fraction 1.0. GPU0/2 processes were unchanged and excluded.

## 2026-09-03 11:40 CST - Reject slow-update challenger and recover 19700 champion

- Context: style02 champion comparison at checkpoint 19800. The slow-update challenger halved policy/discriminator learning rates (`1e-6/2e-6 -> 5e-7/1e-6`) to prevent late AMP/stand drift.
- Challenger training evidence: 19700-19800 had lower KL/clip (`0.00190/0.0105`) and higher update completion (`0.995`), but fixed Isaac playback regressed backward speed to `-0.206 m/s` and retained lateral overshoot at `0.369 m/s`; stand was `0.01 m/s` planar and `0.051 rad/s` yaw. It failed the speed-tracking guardrail despite no terminations.
- Decision: reject slow-update challenger. Restore the best fixed-behavior checkpoint `checkpoint_19700.pt` from `style02_resume19600_20260902` and original style02 parameters (`policy LR=1e-6`, `discriminator LR=2e-6`, `amp_weight=0.02`, PPO epochs=2, smoothing=0.05).
- Recovery: new run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_style02_recover19700_20260903`, tmux `bfm_style02_recover19700_20260903`; no source or reward change. Fresh SSH reconnect proved PID 2785529 on physical GPU1 and progress `19700 -> 19731`; the one spike at 19731 recovered by 19740-19751 with no sustained termination.
- Evaluation evidence retained: `/tmp/eval_isaac19700/` and `/tmp/eval_isaac19800_slow/`; slow challenger is not eligible for promotion. Remaining issues are backward under-response and lateral transient overshoot, requiring representation-level work rather than another blind LR scalar search.

## 2026-09-03 12:10 CST - Recovered champion stable after slow challenger rejection

- Fresh GPU11 discovery found the recovered style02 process `2785529` alive in tmux `bfm_style02_recover19700_20260903`, using physical GPU1 (`CUDA_VISIBLE_DEVICES=1`); GPU0 and GPU2 remained occupied by unrelated jobs.
- The 70-step post-reconnect recovery window reached iteration 19788 with no sustained spike: last-50 means vx/vy/yaw MAE `0.189/0.128/0.230`, stand planar RMS/yaw `0.112/0.149`, PPO update fraction `0.986`, termination `1e-6`, AMP score `-0.084`, discriminator gap `0.081`. The transient at 19731 recovered; no NaN/OOM/NCCL signature.
- Current status: active model is the recovered style02 champion from checkpoint_19700; slow-LR challenger is rejected/tombstoned. No new code or reward change was applied. The next safe mutation requires a representation-level solution for backward/lateral asymmetry, not another scalar LR or lateral-only sampling trial.

## 2026-09-03 18:20 CST - Roll back over-regularized crossaxis15 line to crossaxis5

- Fresh scheduler state advanced `iteration 21709 -> 21953`, `trainlog_iteration` matched, and checkpoint advanced `21700 -> 21900`; the same live PID `2834243` remained reported. Training-only evidence showed AMP score `-0.418 -> -0.381` but discriminator gap remained around `0.56`, reward plateau, `vx` MAE around `0.20`, and no fresh fixed play artifact. The current crossaxis15 lineage had been running far past the last behavior-validated checkpoint.
- Same-lineage command inspection confirmed `cross_axis_velocity_weight=15.0`, `command_backward_scale=1.4`, physical GPU1, and complete `checkpoint_21900.pt` (4,836,627 bytes). GPU0 and GPU2 were unrelated and excluded.
- Causal action: high cross-axis weight was over-regularizing/allowing style discriminator drift; prior fixed Isaac evidence for the same task showed crossaxis5 at checkpoint 20000 reduced forward/lateral cross-axis bleed while preserving stand and backward response. Stopped only the crossaxis15 target, retained rollback checkpoint 21900, and resumed the previously validated crossaxis5 configuration from complete `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_crossaxis5_resume19900_20260903/checkpoint_20000.pt`.
- New run: `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_crossaxis5_resume20000_20260903`, tmux `bfm_crossaxis5_resume20000_20260903`, policy/discriminator settings unchanged except cross-axis weight `15.0 -> 5.0`. Fresh reconnect verified PID `2875837` on physical GPU1, progressing `20000 -> 20019`; latest termination `0`, PPO update `1.0`, cross-axis contribution about `-0.0013` to `-0.0017`.
- Rollback: crossaxis15 checkpoint `checkpoint_21900.pt` and source backup `/tmp/ht_bfm_backup_crossaxis15_to5_20260903_amp.py` retained. The current active model is crossaxis5; the crossaxis15 lineage is not to be rediscovered unless explicitly reconsidered.

## 2026-09-03 12:30 CST - Current champion revalidated after recovery

- Fresh inspection of GPU11 confirms only the recovered style02 Stage2 trainer is the selected HT_BFM lineage on physical GPU1: PID `2785529`, tmux `bfm_style02_recover19700_20260903`, `CUDA_VISIBLE_DEVICES=1`; GPU0 and GPU2 remain unrelated jobs.
- The same lineage has produced a complete `checkpoint_19800.pt` and advanced to iteration `19812`. Last-50 training means are vx/vy/yaw MAE `0.186/0.127/0.232`, stand planar RMS/yaw `0.100/0.149`, command correlations `0.881/0.825`, PPO update fraction `0.988`, termination `2e-6`, AMP score `-0.088`, discriminator gap `0.081`.
- This is operationally healthy and preserves the best fixed Isaac behavior checkpoint (`19700`); no additional mutation is applied. Residual backward/lateral errors remain unresolved and are not being hidden by reward totals.

## 2026-09-03 13:05 CST - Lateral command-scale challenger and expert motion inventory

- Expert dataset used by the current Stage2 run: `dataset/pkl_cmd/piplus_lse_balanced5_plus_turn4_weighted.pkl` with 14 entries. Entries 0-9 are two copies each of the five LSE motions: `walk_ff_loop_360_001__A049`, `neutral_walk_ff_360_R_002__A534`, `default_pose_piplus_s_lse_stand`, `B4 Stand_to_Walk_backwards_stageii_piplus_s_LSE`, and `B5 Walk_backwards_stageii_piplus_s_LSE`. Entries 10-13 are `balanced_lateral_left`, `balanced_lateral_right`, `balanced_yaw_left`, and `balanced_yaw_right`; the added 50 Hz motions carry approximate commands left `[0.110,0.229,-0.043]`, right `[-0.056,-0.234,-0.200]`, yaw-left `[0.111,0.050,0.373]`, yaw-right `[-0.013,0.097,-0.297]`.
- Evidence before mutation: default lateral encoder scale `5.0` produced about `0.317-0.359 m/s` for a `0.25 m/s` lateral command on fixed Isaac screens; scale `4.0` produced `0.238 m/s` at checkpoint 19700 but drifted to `0.332 m/s` at checkpoint 19800 and introduced cross-axis yaw `-0.224 rad/s`. Scales `3.5` and `2.5` under-tracked at the 300-step endpoint (`0.178` and `0.139 m/s`).
- Implementation: added optional `--command-lateral-scale` to Stage2 training/play, stored the effective scale in checkpoint metadata, and added validation/tests. Remote backups: `/tmp/ht_bfm_backup_lateral_scale_20260903_123500_amp.py`, `_play.py`, `_test.py`; applied diffs: `/tmp/ht_bfm_lateral_scale_applied_amp.patch`, `_play.patch`, `_test.patch`.
- Challenger: scale `4.0`, policy optimizer reset, resumed from complete `checkpoint_19700.pt`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_lateral4_resume19700_20260903`, tmux `bfm_lateral4_resume19700_20260903`. It reached checkpoint 19800 with no terminations, but fixed Isaac forward/backward/stand/yaw remained acceptable while lateral was `0.332 m/s` and yaw cross-axis `-0.224 rad/s`; rejected and never promoted.
- Recovery: restored original scale `5.0` from checkpoint 19700 in `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_style02_recover19700_20260903b`, tmux `bfm_style02_recover19700_20260903b`, PID `2803801`. Fresh reconnect confirmed iteration `19700 -> 19714`, termination 0 and stable early MAE. Local validation: 23 tests passed and Ruff passed; remote env Python tests: 21 passed (system Python lacks mujoco).

## 2026-09-03 14:10 CST - Lateral-scale implementation and challenger decision

- The optional lateral command preprocessing scale is now implemented in both remote training and playback code, with effective scale recorded in `encoder_input_transform.command_scale`; default behavior remains scale `5.0`.
- Formal fixed Isaac screens on the 19700 policy gave lateral endpoint velocities: scale 5.0 `~0.317 m/s`, 4.0 `~0.238 m/s` at 300 steps on the isolated screen but `~0.332 m/s` after 100 training updates with yaw cross-axis `-0.224 rad/s`, 3.5 `~0.178`, and 2.5 `~0.139`. Scale 4.0 challenger was rejected for cross-axis coupling and no robust full-direction improvement.
- Recovered original scale 5.0 and the best fixed-behavior checkpoint 19700. Current active run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_style02_recover19700_20260903b`, tmux `bfm_style02_recover19700_20260903b`, PID `2803801`, is alive on physical GPU1 after independent reconnect; latest observed iteration 19780, last-50 MAE `0.193/0.130/0.234`, stand RMS/yaw `0.113/0.150`, termination ~`1e-6`, PPO update `0.982`.
- No further scalar sweep is authorized by the evidence. Remaining backward/lateral residuals likely require a command-conditioned representation or action-routing change, which must be designed and evaluated as a separate architecture challenger.

## 2026-09-03 15:10 CST - Restore scale5 champion after lateral4 rejection

- Formal lateral-scale screening used the project-native Isaac evaluator on checkpoint 19700. Scale 4.0 improved one isolated endpoint but did not survive the mature 19800 screen: lateral settled at `0.332 m/s` for a `0.25 m/s` target and induced yaw cross-axis `-0.224 rad/s`; scale 3.5/2.5 under-tracked. The representation challenger is rejected.
- Remote backups before the input-scale implementation: `/tmp/ht_bfm_backup_lateral_scale_20260903_123500_amp.py`, `_play.py`, `_test.py`; reversible diffs: `/tmp/ht_bfm_lateral_scale_applied_amp.patch`, `_play.patch`, `_test.patch`. Local `py_compile`, 23 unit tests, and Ruff all pass; remote environment tests pass 21/21.
- Restored scale5 champion from complete checkpoint 19700 in `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_style02_recover19700_20260903b`, started new tmux `bfm_style02_recover19700_20260903b` with PID `2803801`, and verified through a fresh JumpServer reconnect. It subsequently produced checkpoint_19800 (4,836,499 bytes) and is currently alive at iteration 19812 on physical GPU1; recent last-50 metrics remain finite with vx/vy/yaw MAE `0.186/0.127/0.232`, stand RMS/yaw `0.100/0.149`, PPO update `0.988`, termination `~1e-6`, AMP gap `~0.081`.
- Current status: one active champion, no duplicate Stage2 worker, no new scalar mutation pending. The expert inventory remains the 14 weighted motions listed above; remaining failures are backward under-response and lateral transient/cross-axis coupling.

## 2026-09-03 16:40 CST - Accept backward command representation at 19900

- Evidence before mutation: on checkpoint 19800, a negative-vx encoder multiplier of `1.4` improved the fixed Isaac backward endpoint from about `-0.284` to `-0.313 m/s` with yaw cross-axis about `-0.013 rad/s`; multiplier `1.6` regressed to `-0.249 m/s`. The candidate was therefore limited to a single 1.4 representation change with policy optimizer reset.
- Challenger `backward14_resume19800_20260903` reached checkpoint 19900 with no safety failure. Fixed Isaac screen (`/tmp/eval_isaac_backward14_19900/`, seed 1, 300 steps, stand/forward/backward/lateral/yaw) completed all five scenarios without termination: stand `~0.001 m/s, 0.025 rad/s`, forward `0.391 m/s`, backward `-0.325 m/s`, lateral `0.288 m/s`, yaw `0.557 rad/s`. This improved backward and lateral response over the old 19800 screen while retaining stable stand/turn behavior.
- Promoted 19900 as the current behavior champion and resumed unchanged 1.4 configuration from complete `checkpoint_19900.pt` in `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_backward14_recover19900_20260903`, tmux `bfm_backward14_recover19900_20260903`. Fresh reconnect proved PID `2823081` alive on physical GPU1, advancing `19900 -> 19934`; latest MAE `0.161/0.117/0.227`, stand RMS/yaw `0.065/0.121`, PPO update 1.0, termination 0.
- Checkpoint 20000 was screened before promotion: it retained backward `-0.330 m/s` but developed forward/lateral cross-axis velocities (`vy=-0.113`, lateral `vx=-0.089,wz=-0.160`), so 19900 remains selected and 20000 is not promoted.

## 2026-09-04 11:00 CST - Action-imitation regression requires remote discriminator evaluation

- Context: Active GPU11 PiPlus Stage2 lineage `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_crossaxis5_resume20000_20260903`, latest inspected checkpoint `checkpoint_30200.pt`.
- Phenomenon: The latest readable training row near iteration 30272 shows poor AMP/style alignment (`amp_score≈-2.97`, discriminator score gap `≈2.05`) while direct command tracking remains finite (`planar MAE≈0.202`, `vx/vy/yaw MAE≈0.153/0.103/0.208`) and termination/crash/fall rates are zero.
- Analysis: This pattern is consistent with discriminator saturation or drift against the weighted 14-motion expert distribution, not an immediate locomotion-safety collapse. Prior conditional-AMP and scalar-only challengers regressed command tracking, so an AMP-weight increase alone is not justified. A discriminator-reset/保守 adversarial-resume challenger should be compared with fixed Isaac headless playback before adoption.
- Adjustment: No remote parameter or source change applied yet; GPU11 JumpServer asset currently returns connection refused, so process/GPU/checkpoint compatibility and fixed-play evidence cannot be revalidated safely.
- Rationale: Avoid stopping or resuming an unverified lineage and avoid blind reward/LR changes while the required remote evidence channel is unavailable.
- Expected effect: Pending remote access; next cycle should run the fixed five-command Isaac protocol, then test only a checkpoint-compatible discriminator-reset/low-pressure AMP change if style failure is confirmed without tracking regression.
- Result: Pending external GPU11 connectivity and remote evaluation.
- Files/commands: `humanoidverse/amp_stage2.py` (read-only review); remote lineage and logs require fresh JumpServer connection.

## 2026-09-04 13:12 CST - Conservative discriminator-reset resume

- Context: PiPlus Stage2 `crossaxis5_resume20000_20260903`, GPU11 physical GPU1; source checkpoint `checkpoint_31100.pt` and derived `checkpoint_31100_discreset.pt`.
- Phenomenon: Equal training windows 20000-20100 vs 31000-31147 degraded AMP score `-0.163 -> -3.426`, discriminator gap `0.126 -> 2.365`, and planar MAE `0.177 -> 0.208`, while termination stayed near zero. Fixed Isaac checkpoint 31100 had no termination across stand/forward/backward/lateral/yaw.
- Analysis: The discriminator had saturated or drifted against the weighted 14-motion expert distribution. A reset-only challenger restored AMP metrics but its checkpoint 31400 fixed screen regressed forward/backward/lateral endpoints, so adversarial pressure also needed reduction.
- Adjustment: Replaced only the 202-D discriminator and removed its optimizer state; changed AMP weight `0.02 -> 0.01` and policy LR `1e-6 -> 5e-7`; all robot, action, observation, dataset, and cross-axis contracts were preserved.
- Result: New amp001 lineage resumed from iteration 31100 and reached 31155 after reconnect with AMP score `-0.102`, discriminator gap `0.027`, AMP reward contribution `0.024`, PPO update fraction `1.0`, and termination `0`.
- Resume: tmux `bfm_discreset_amp001_20260904`, PID `3126231`, work dir `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_discreset_amp001_from31100_20260904`, `CUDA_VISIBLE_DEVICES=1`; old reset-only challenger was stopped and is not active.
- Persistence: launch connection closed, waited 40 seconds, and a fresh JumpServer/GPU11 connection found the same tmux/PID/workdir on physical GPU1 with progress `31116 -> 31155`; unrelated GPU0/GPU2 jobs were unchanged.
- Rollback: original and derived checkpoints remain in the original run directory; no source file patch was applied.

## 2026-09-04 14:55 CST - amp001 monitoring: early style recovery, late drift

- Context: Active `discreset_amp001_from31100_20260904`, tmux `bfm_discreset_amp001_20260904`, GPU11 physical GPU1.
- Observation: Training remains live through `checkpoint_32000.pt` and iteration about 32061. AMP score improved from about `-3.4` before reset to `-0.10` in 31100-31200, then drifted to `-0.436` in 31900-32000 and `-0.485` in 32000-32055; discriminator gap rose from `0.025` to `0.252`.
- Task guardrails: recent training planar/vx/vy/yaw MAE are approximately `0.258/0.211/0.112/0.225`, termination remains zero, PPO update fraction is about `0.997`, and KL/value loss remain finite.
- Analysis: Expert-style alignment is materially better than the pre-reset saturated lineage but is not monotonically improving; classify the challenger as improving-then-regressing/pending. Do not increase AMP pressure blindly; the next decision requires a fresh fixed Isaac screen against checkpoint 31100.
- Result: Monitoring only; no new parameter change or restart.

## 2026-09-04 15:20 CST - Continuous monitor: amp005 remains healthy

- Context: Active `discreset_amp005_from31200_20260904`, tmux `bfm_discreset_amp005_20260904`, PID `3153578`, physical GPU1.
- Observation: Fresh GPU11 reconnect found the same process and tmux alive; log reached iteration 31266 with AMP score `0.0283`, discriminator gap `0.0018`, PPO update fraction `1.0`, KL `9.4e-5`, termination `0`, and no fatal/non-finite signature. Checkpoint 31200 is complete; next save is pending.
- Analysis: The conservative reset (`amp_weight=0.005`, policy LR `1e-7`) is currently preventing the severe AMP saturation and policy collapse seen in the prior lineage. This is an early improving window, not yet a mature acceptance result; continue monitoring later checkpoint-aligned Isaac screens, especially forward/backward/lateral tracking.
- Result: No new mutation or restart; leave amp005 training running.

## 2026-09-04 15:45 CST - amp005 checkpoint31300 fixed Isaac screen

- Context: Active `discreset_amp005_from31200_20260904`, checkpoint `checkpoint_31300.pt`, GPU11 physical GPU1; evaluation isolated to GPU0 with the project-native `humanoidverse.amp_stage2_play` Isaac headless protocol, 300 steps per command.
- Result: All five scenarios completed without termination. Endpoint velocities: stand `[-0.002,0.008,-0.034]`, forward `[0.308,-0.004,0.028]`, backward `[-0.114,-0.126,0.130]`, lateral `[-0.078,0.179,-0.010]`, yaw `[-0.067,0.034,0.612]`.
- Analysis: Stand/forward/yaw are stable and forward response is close to target; backward and lateral remain underfit with cross-axis coupling. Do not repeat historical lateral-only oversampling (0.15) because it caused delayed yaw collapse. Keep amp005 pending and continue collecting checkpoint-aligned evidence.
- Training status: latest observed iteration 31331, AMP score `0.016`, discriminator gap `0.009`, PPO update fraction `1.0`, termination `0`.
- Action: No new mutation or restart; leave `bfm_discreset_amp005_20260904` running.

## 2026-09-04 16:40 CST - freshopt amp005 early monitoring

- Context: Active remote lineage `freshopt_amp005_from31300_20260904`, tmux `bfm_freshopt_amp005_20260904`, physical GPU1.
- Observation: Fresh policy/discriminator optimizer reset resumed from checkpoint 31300. At iteration 31364, AMP score is `0.062`, discriminator gap `-0.012`, PPO update fraction `1.0`, KL `4.5e-5`, with no termination/fatal signature.
- Analysis: Resetting both optimizers removes the accumulation that preceded the 31500 action drift in the prior amp005 run; style alignment is currently near expert/policy score parity. This remains an early window pending a complete checkpoint and fixed Isaac screen.
- Result: No additional mutation; keep the single freshopt challenger running and monitor the next checkpoint.

## 2026-09-04 16:10 CST - amp005 checkpoint31500 monitoring

- Context: Active `discreset_amp005_from31200_20260904`, checkpoint `checkpoint_31500.pt`, tmux `bfm_discreset_amp005_20260904`, physical GPU1.
- Observation: Fresh GPU11 inspection found the same PID/tmux alive at iteration about 31527. AMP score remained near zero (`-0.010` mean in 31400-31500; latest about `-0.031`), discriminator gap `0.026-0.036`, PPO update fraction `1.0`, termination `0`, and no NaN/OOM/NCCL/fatal signature. Equal windows show planar MAE `0.222 -> 0.236` and yaw MAE `0.226 -> 0.229` from 31200-31300 to 31400-31500.
- Analysis: Discriminator reset plus low AMP pressure prevents the prior saturation/collapse, but tracking has a small drift. The window is not sufficient to justify another mutation; keep the current challenger pending and evaluate the next mature checkpoint with fixed Isaac before changing parameters.
- Result: No new mutation or restart; training left running.

## 2026-09-04 19:05 CST - MPC-enabled single-GPU resume persistence check

- Context: Video-matched `checkpoint_25300.pt` resumed in `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_mpc4_resume25300_20260904`; tmux `bfm_mpc4_1gpu_resume25300_20260904`, physical GPU1.
- Persistence: After closing the launch SSH and waiting 40 seconds, a fresh JumpServer/GPU11 connection found PID `3185470` and the same tmux/work directory alive; progress advanced from iteration `25300` to about `25433`, with checkpoint `25400.pt` complete.
- MPC evidence: `world_model_mpc_ready=1.0`, MPC control fraction approximately `0.18-0.19`, JEPA online motion MAE approximately `0.10`, planner distill loss approximately `0.017`, and no fatal/NaN/OOM/NCCL signature.
- Performance snapshot: AMP score about `0.15-0.19`, discriminator gap about `0.66-0.68`, planar MAE about `0.25-0.32`, yaw MAE about `0.49-0.57`, termination near zero. This is an early post-resume window and requires mature fixed Isaac comparison.
- Deployment note: Removing MPC at deployment is not guaranteed equivalent because MPC-controlled transitions are excluded from PPO and distilled into the command encoder; run a separate no-MPC playback screen with the same checkpoint before hardware use.
- Result: MPC-enabled training is active and persistent; no parameter mutation applied in this check.

## 2026-09-04 18:25 CST - Resume MPC/JEPA lineage from video checkpoint 25300

- Context: User identified `stage2_playback_piplus_jepa_mpc4_25300_gamepad.mp4`; local Isaac log maps it to `logs/amp_stage2_piplus_lse_2gpu_4096env_1m_mpc4_rankbalanced_v2_resume19900_20260806/checkpoint_25300.pt`.
- Checkpoint evidence: complete 50,744,246-byte checkpoint, stable size/mtime, SHA256 `ab036ebbf5e6c8ec5ed9496b6dfc83e7e84763b781edb63fcf86ecbdedf358ff`; contains policy/discriminator/AMP normalizer plus `world_model`, `world_model_optimizer`, and `world_model_runtime`; strict JEPA state load passed.
- MPC contract: world model observation/action/context/embedding `631/23/8/256`; MPC horizon 5, candidates 4, residual scale 0.15, control fraction 0.25, max envs 1024, distill coefficient 0.2. Metadata reports `world_model_mpc_ready=true`.
- Launch: initial 2-GPU attempt stopped after NCCL duplicate-GPU conflict and no iteration progress; same checkpoint and MPC parameters were relaunched safely on idle physical GPU1 as a single-GPU 4096-env run to avoid an unrelated GPU0 job.
- Active resume: `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_mpc4_resume25300_20260904`, tmux `bfm_mpc4_1gpu_resume25300_20260904`, PID `3185470`; source `/tmp/codex_upload_mpc4_20260806_2227/amp_stage2.py`, `CUDA_VISIBLE_DEVICES=1`.
- Early result: iteration 25305 has AMP score `0.190`, discriminator gap `0.681`, planar/vx/vy/yaw MAE `0.226/0.148/0.142/0.445`, termination `0`, MPC ready `1.0`, MPC control fraction `0.184`, JEPA motion MAE `0.100`, and planner distill loss `0.0178`. Keep pending until checkpoint-aligned evaluation after warmup.
- Deployment note: training with MPC is supported; deployment without MPC must use the non-MPC entry and be separately evaluated because MPC-controlled transitions are excluded from PPO and distilled after the fit gate.

## 2026-09-05 12:20 CST - Reward-tracking/upright resume from checkpoint 32100

- Context: The user-reported `checkpoint_32100.pt` from `amp_stage2_piplus_lse_1gpu_4096env_1m_mpc4_resume25300_20260904` was evaluated with the project-native pure-policy Isaac screen. All five commands completed without termination, but stand/forward/backward/lateral/yaw endpoints were approximately `[0.003,-0.001,0.036]`, `[-0.001,0.002,0.031]`, `[-0.476,-0.026,-2.311]`, `[-0.010,-0.003,-0.009]`, and `[0.000,-0.061,0.316]`; direct forward/lateral response was nearly absent and backward had severe yaw coupling.
- Training evidence: The old lineage had drifted to planar/vx/vy/yaw MAE around `0.277/0.206/0.148/0.686-0.75`, with AMP score around `0.10-0.11`, discriminator gap around `0.79`, termination `0`, and MPC control fraction about `0.20`.
- Reward adjustment: `linvel_exp 1.4 -> 2.4`, `linvel_projection 0.0 -> 0.4`, `angvel_z_exp 1.0 -> 1.8`, `backward_velocity_progress 1.5 -> 1.8`, `turn_rate_progress 0.0 -> 0.4`, `body_upright 1.5 -> 2.4`, and `stand_still 2.1 -> 1.5`. This raises dense command/upright gradients while reducing the over-dominant generic stand-still term.
- Optimization/MPC adjustment: deployment-matched command smoothing `0.02 -> 0.10`; MPC control fraction `0.25 -> 0.10`; planner distillation `0.2 -> 0.5`; AMP weight `0.25 -> 0.15`; PPO epochs `5 -> 3`, policy LR `2.1e-5 -> 5e-6`, target KL `0.04 -> 0.02`, discriminator LR `1e-5 -> 5e-6`. The policy optimizer was reset while policy/discriminator/world-model states and world-model optimizer/runtime were retained from checkpoint 32100.
- Reproducibility: source backup `/tmp/amp_stage2_rewardtrack_upright_20260905.before.py`; candidate `/tmp/amp_stage2_rewardtrack_upright_20260905.py` (SHA256 `b837edb01e8628946929e59eda77a10ed830658c6b8f0fc9eaef45a62d679c41`); patch `/tmp/amp_stage2_rewardtrack_upright_20260905.patch` (SHA256 `ba67ce4185e8d7ea0a1cdfa3eaad1d793fb3a1ab924f58297c39a97c52d96f33`); derived resume `/tmp/amp_stage2_rewardtrack_upright_resume32100.pt` (SHA256 `f02c79f968b4396d969fb877f73ef9041e6970f5190fd39de72bc89e14646e3e`).
- Resume: run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_upright_resume32100_20260905`, tmux `bfm_rewardtrack_upright_resume32100_20260905`, PID `3416732`, physical GPU1. A fresh JumpServer/GPU11 reconnect verified progress through iteration `32117`, MPC ready, no termination/fatal signature, KL about `0.0065-0.0089`, and clip fraction about `0.08-0.13`.
- Early result: tracking remains noisy during optimizer reset, but the last observed row had planar/vx/vy/yaw MAE `0.277/0.207/0.147/0.715`, termination `0`, and world-model motion MAE `0.099`. Keep this challenger pending until checkpoint 32200 is complete and screened with the same fixed pure-policy protocol.
- Checkpoint screen: complete `checkpoint_32200.pt` (50,743,990 bytes) was tested with the same five fixed Isaac commands on isolated GPU0. Endpoints were stand `[0.003,-0.002,0.022]`, forward `[-0.001,0.002,0.027]`, backward `[-0.178,0.035,-2.278]`, lateral `[-0.001,-0.003,-0.004]`, and yaw `[-0.048,-0.124,0.333]`, all with `terminated=False`. The short 100-update screen is therefore not an acceptance result; keep the reward challenger running for a longer window rather than promoting 32200.
- Current persistence: the same GPU1 worker remains alive after the evaluation and has advanced to about iteration `32334`, with complete `checkpoint_32300.pt`, termination `0`, planar/vx/vy/yaw MAE about `0.257/0.188/0.140/0.656`, AMP score `0.087`, discriminator gap `0.813`, and MPC ready `1.0` at control fraction `0.078`.

## 2026-09-05 17:20 CST - Pull latest reward-track checkpoint locally

- Remote run inspection through the JumpServer direct asset user confirmed the same reward/upright trainer is still active on GPU11 physical GPU1 (PID `3416732`, tmux `bfm_rewardtrack_upright_resume32100_20260905`). The newest complete artifact at inspection was `checkpoint_34500.pt`, 50,743,990 bytes; the console had reached iteration `34424` with termination `0`, planar/vx/vy/yaw MAE `0.196/0.127/0.126/0.417`, and MPC ready `1.0`.
- Downloaded to `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_upright_resume32100_20260905/checkpoint_34500.pt`. Local and remote SHA256 both equal `c335ff4152bba8e73d76dfa159116842c07d078cc0fcfd1809ed7a84b3dffc34`.
- Updated the related playback/export commands in `docs/play_checkpoint_bfmzero-piplus-lse.md` to this local run/checkpoint. The remote trainer was not stopped or modified.

## 2026-09-06 - Smooth-stand/imitation/tracking challenger from checkpoint 36300

- Context: Active reward-track/upright run reached complete `checkpoint_36300.pt`; fixed Isaac screen remained termination-free with stand `[0.007,0.002,0.006]`, forward `[0.004,0.012,-0.068]`, backward `[-0.372,-0.186,0.050]`, lateral `[0.158,0.051,-0.179]`, and yaw `[0.065,-0.075,0.682]`. Training logs still showed high smoothness diagnostics (action-rate2 about `0.008`, joint acceleration about `2.0e4`, stand-still raw about `0.46`) and a large discriminator gap around `0.69-0.71`.
- Adjustment: created `/tmp/amp_stage2_smoothstand_20260906.py` from the active source, with `stand_still 1.50 -> 2.40`, `action_rate_l2 0.05 -> 0.07`, `action_rate2_l2 0.10 -> 0.16`, `joint_acc_l2 9.2e-7 -> 1.3e-6`, `joint_vel_l2 4.2e-3 -> 5.0e-3`, `energy_l1 5e-4 -> 6e-4`, `angvel_xy_l2 0.135 -> 0.17`, and `body_upright 2.4 -> 2.7`; command overrides `linvel_exp 2.4 -> 2.8`, `backward_velocity_progress 1.8 -> 2.0`, `turn_rate_progress 0.4 -> 0.55`, `linvel_projection 0.4 -> 0.5`; AMP weight `0.15 -> 0.20`. Robot/action/observation/dataset contracts are unchanged.
- Rationale: strengthen zero-command pose/torso and first/second-difference damping while retaining direct velocity and expert-style gradients; reset only policy optimizer state to remove stale update momentum from the preceding jittery window.
- Reproducibility: backup `/tmp/amp_stage2_smoothstand_20260906.before.py`; patch `/tmp/amp_stage2_smoothstand_20260906.patch` (SHA256 `f7589fd53e10c8d942cce57033e71b2055285ab10775575ce28cb686b6c715e2`); candidate SHA256 `caa3013675765d22c9a78f773788476155bdf3a07c56b7299a66e3f5ce614295`; derived resume `/tmp/amp_stage2_smoothstand_resume36300.pt` SHA256 `465dff43bf5eff2b1540a13c99dbb9f79026bf6f78ece6d8c441dd36e919acdd`.
- Resume: `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_smoothstand_imitation_tracking_resume36300_20260906`, tmux `bfm_smoothstand_imitation_tracking_resume36300_20260906`, PID `3572637`, `CUDA_VISIBLE_DEVICES=1`; remote Stage2 tests passed 21/21 and candidate py_compile passed. Independent post-launch connection verified iteration `36300 -> 36308`, MPC ready `1.0`, termination `0`.
- Result: `challenger_pending`; first row after reset had smoothness raw action-rate2 `0.00684`, joint acceleration `1.67e4`, stand-still `0.251`, planar/vx/vy/yaw MAE `0.291/0.202/0.169/0.366`, AMP score `0.144`, discriminator gap `0.700`. Do not promote before a mature checkpoint-aligned fixed screen.

## 2026-09-06 - Checkpoint 36400 smooth-stand challenger screen

- Evaluation: complete `checkpoint_36400.pt` was run through the identical five-command Isaac protocol on isolated physical GPU0; all scenarios finished with `terminated=False`. Endpoints: stand `[0.003,-0.002,0.004]`, forward `[0.005,-0.006,-0.022]`, backward `[-0.456,0.064,-0.065]`, lateral `[0.078,0.130,-0.195]`, yaw `[0.027,-0.065,0.799]`.
- Analysis: zero-command posture is cleaner and lateral/backward cross-axis is reduced relative to 36300, while forward remains effectively unresponsive and yaw/backward are somewhat over-driven. This is an early checkpoint after the optimizer reset, so it is `challenger_pending`, not promotion or rejection.
- Current continuation: the same candidate remains alive on GPU1, with the post-reconnect log advancing through iteration `36448`; latest observed termination remains `0`. Continue to the declared mature window before deciding whether to reduce reverse/turn progress or address forward command representation.

## 2026-09-06 - Smooth-stand challenger persistence after evaluation

- The 36400 fixed screen completed without termination in all five scenarios. After evaluation cleanup, the same tmux/PID remained alive on physical GPU1 and advanced to iteration `36514`; no evaluation worker remained. This confirms the challenger survived the independent evaluation run and continues from the 36300 resume lineage.
- The challenger remains pending: stand and lateral posture improved, but fixed forward response is still near zero and reverse/yaw endpoints are somewhat aggressive. No additional parameter stack was applied before the mature-window decision.

## 2026-09-06 - Resume PiPlus H0W fixed motion-matched box from model-only checkpoint

- Context: User requested the BFM box environment to inherit InstinctLab shadowing initialization: motion-local terrain and motion reference share one terrain origin; no post-reset box-edge translation. The source checkpoint is `/home/sunteng/Project/HT_BFM/huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)` and contains model weights only.
- Changes: Added `data_process/build_piplus_h0w_box_bfm.py` for the 363-frame MuJoCo-qpos clip, `external_mesh_terrain.py` integration for the paired OBJ, `bfm_zero_piplus_h0w_box.yaml`, fixed frame-zero reset, and model-only warm-start support in `humanoidverse/train.py`. Box config disables push, COM/mass/friction/PD/default-pose/control-delay/reset randomization, motion-start randomization, initial noise, and lie-down reset.
- Validation: Box motion pkl has one 22-DoF motion at 30 FPS; remote model load succeeded as `FBcprAuxModel`, `z_dim=256`, actor input 360. IsaacLab scene generation and external box terrain import completed. Two short Isaac play attempts were stopped after Vulkan device-lost on shared evaluation GPUs; no training lineage was touched.
- Remote: Isolated project `/root/autodl-tmp/zhuzejian/Project/HT_BFM_box_origin_20260905`; persistent tmux supervisor `bfm-box-origin-gpu11-20260906`; target physical GPU2 (GPU11). It waits until GPU2 is below 8 GB and 10% utilization, then runs `scripts/resume_box_origin_gpu11.sh` with `--robot PiPlus_S_12L8A0G2H0W --box_climb --warm_start_model_from .../bootstrap --no_resume_replay_buffer --work_dir results/bfmzero-piplus-h0w-box-origin-20260906`.
- Current status: Supervisor was independently verified alive; GPU2 remains occupied by unrelated UID 1005 `zhangrui/ufo_walk`, so training has not started and no unrelated process was stopped.
- Launch repair: added `--no_wandb` because this isolated environment has no configured W&B API key; local run logs remain authoritative. The supervisor script now passes this flag and was syntax/py_compile validated remotely.

## 2026-09-06 12:00:52 CST - Speed-tracking reward resume from checkpoint 42000

- Context: GPU11 HT_BFM smoothstand lineage; tmux bfm_speedtrack_from42000_20260906; physical GPU1
- Phenomenon: User reports checkpoint_42900 speed tracking degradation. Training window near 42900 had direct vx MAE about 0.17-0.18 but fixed seed=1 forward playback showed velocity [-0.252, 0.043, -6.759], while checkpoint_42000 showed [0.043, 0.360, -1.356]; termination remained false.
- Analysis: The weighted upright contribution (~0.059) exceeded direct linvel contribution (~0.042), while projection contributed only ~0.0006, so speed-command gradients were underweighted relative to posture/AMP pressure. Fixed playback is noisy, but 42000 is the least-regressed compatible checkpoint among tested candidates.
- Adjustment: Derived /tmp/amp_stage2_speedtrack_resume42000.pt from checkpoint_42000.pt with policy_optimizer reset; reward candidate: linvel_exp 2.80 -> 3.40, linvel_projection 0.50 -> 1.00, body_upright 2.70 -> 2.30, stand_still 2.40 -> 1.90; launch override --linvel-exp-weight 3.4.
- Rationale: Increase direct velocity and signed projection gradients while reducing competing upright/zero-command pressure; reset policy Adam moments to avoid stale momentum under changed reward scales.
- Expected effect: Lower nonzero vx/vy tracking MAE and improve command-response slope/correlation without increasing termination or high-frequency smoothness costs; monitor AMP gap and PPO KL.
- Result: Pending mature checkpoint-aligned window; fresh reconnect verified the new process advanced 42000 -> 42010 with termination_rate=0.
- Files/commands: Remote backup /tmp/amp_stage2_speedtrack_20260906.before.py; patch /tmp/amp_stage2_speedtrack_20260906.patch; candidate humanoidverse/amp_stage2_speedtrack_20260906.py; resume hash 558b5c8020abd90e21a84624c39c21b0d8e5866e2d1962a6927d6832bdd2d743; tmux bfm_speedtrack_from42000_20260906; CUDA_VISIBLE_DEVICES=1

## 2026-09-06 12:02:22 CST - Speedtrack candidate early post-resume check

- Context: bfm_speedtrack_from42000_20260906, GPU11 physical GPU1
- Phenomenon: After policy optimizer reset, candidate reached iteration 42024; first 25 rows show nonzero vx MAE 0.255, vy MAE 0.179, yaw MAE 0.282, vx response slope 0.524, correlation 0.825, termination 0.
- Analysis: Early window is noisy but direct linvel contribution is ~0.052 versus upright ~0.050 under the new weights, confirming the intended reward rebalance; no safety regression yet.
- Adjustment: No additional change; continue to checkpoint-aligned mature window.
- Rationale: Avoid stacking a second mutation before the reward change has enough samples.
- Expected effect: Assess 100-500 iteration windows for lower vx/vy MAE and preserved smoothness/termination.
- Result: Pending
- Files/commands: Remote /tmp/speedtrack_from42000_launch.log; tmux bfm_speedtrack_from42000_20260906

## 2026-09-06 12:03:14 CST - Speedtrack candidate validation gate

- Context: bfm_speedtrack_from42000_20260906
- Phenomenon: Reward candidate source and resume launched on GPU11 before mature evaluation.
- Analysis: Remote project unit tests cover Stage 2 reward/checkpoint behavior and pass; candidate module also passed py_compile.
- Adjustment: No change.
- Rationale: Establish a clean validation baseline before interpreting training metrics.
- Expected effect: No functional regression from reward-weight edits; continue monitoring live run.
- Result: Remote unittest tests/test_amp_stage2.py: 21/21 passed; py_compile passed; process remains alive at iteration 42024.
- Files/commands: /tmp/amp_stage2_speedtrack_20260906.patch; humanoidverse/amp_stage2_speedtrack_20260906.py

## 2026-09-06 12:05:44 CST - Speedtrack candidate continued monitoring

- Context: bfm_speedtrack_from42000_20260906, fresh GPU11 reconnect
- Phenomenon: Live candidate advanced to iteration 42051 with 52 logged updates; 42000-42051 mean nonzero vx/vy/yaw MAE 0.2563/0.1798/0.2891, vx response slope 0.5264, correlation 0.8168, termination 0.
- Analysis: No safety regression and the intended reward contribution rebalance persists, but the window is still too short after optimizer reset to promote or reject the candidate.
- Adjustment: No additional tuning; continue the same lineage.
- Rationale: Avoid stacking changes before a checkpoint-aligned mature comparison.
- Expected effect: Collect at least one complete post-change checkpoint and compare against the 42000 baseline and 42900 degradation report.
- Result: Pending mature evaluation.
- Files/commands: /tmp/speedtrack_from42000_launch.log; tmux bfm_speedtrack_from42000_20260906

## 2026-09-06 12:14:24 CST - Speedtrack checkpoint 42100 evaluation

- Context: bfm_speedtrack_from42000_20260906, checkpoint_42100, GPU11 physical GPU1
- Phenomenon: Complete checkpoint_42100 is available. Post-change window 42100-42121 (22 updates) has vx MAE 0.2460, vx response slope 0.5648, correlation 0.8357, vy MAE 0.1815, yaw MAE 0.2880, termination 0; baseline 42000 window was vx MAE 0.2557, slope 0.532, correlation 0.818.
- Analysis: The direct velocity reward rebalance is showing an early improvement in vx error and response slope without a safety regression; fixed single-seed playback remains noisy and is not sufficient to reject the candidate.
- Adjustment: No additional change; keep the speedtrack candidate running beyond checkpoint 42100.
- Rationale: Avoid overreacting to one noisy fixed rollout while the aligned training metrics improve.
- Expected effect: Confirm the improvement persists through a 100+ update mature window and remains compatible with smoothness/AMP metrics.
- Result: Promising but pending mature acceptance.
- Files/commands: Remote checkpoint_42100.pt; /tmp/speedtrack_from42000_launch.log; /tmp/eval_speedtrack_42100_fwd_s1.log

## 2026-09-06 12:16:02 CST - Pull speedtrack checkpoint 42100

- Context: bfm_speedtrack_from42000_20260906
- Phenomenon: Checkpoint 42100 completed and candidate remains live at iteration 42135.
- Analysis: The candidate has a reproducible local artifact while remote training continues; 42100 is currently the best checkpoint in this adjusted lineage based on the early aligned window.
- Adjustment: Pulled checkpoint_42100.pt and updated related playback/export docs to the speedtrack run.
- Rationale: Keep the current best compatible checkpoint available for playback and rollback.
- Expected effect: Enable local validation and preserve a concrete resume point if later windows regress.
- Result: Remote/local SHA256 match: 4fcb233b1dfb52f3c029a936c36d44fdfbf29e5d70302ce0a23df5eda2b373e5; docs refreshed.
- Files/commands: logs/amp_stage2_piplus_lse_1gpu_4096env_1m_speedtrack_from42000_20260906/checkpoint_42100.pt; logs/remote_launch/speedtrack_from42000_launch_20260906.log; docs/play_checkpoint_bfmzero-piplus-lse.md

## 2026-09-06 12:22:02 CST - Speedtrack checkpoint 42100 mature-window continuation

- Context: bfm_speedtrack_from42000_20260906, GPU11 physical GPU1
- Phenomenon: Fresh remote polling reached iteration 42180; recent points show vx MAE typically 0.231-0.242, response slope 0.55-0.60, correlation 0.81-0.87, planar MAE about 0.21-0.25, termination 0.
- Analysis: The reward rebalance continues to preserve zero termination and generally improves vx response slope/correlation versus the pre-change 42000 window, though short-point variance remains substantial.
- Adjustment: No additional reward or optimizer change; keep candidate lineage active.
- Rationale: A second mutation would confound the improving speed-tracking signal.
- Expected effect: Continue through checkpoint 42200+ for a stronger acceptance/rejection decision.
- Result: Promising, not yet final acceptance.
- Files/commands: Remote /tmp/speedtrack_from42000_launch.log; checkpoint_42100.pt local copy; tmux bfm_speedtrack_from42000_20260906

## 2026-09-06 12:34:58 CST - Cross-axis stability correction from checkpoint 42200

- Context: bfm_speedtrack_stable_from42200_20260906, GPU11 physical GPU1
- Phenomenon: Checkpoint_42200 fixed tests improved forward velocity to [0.164,-0.239,-0.196] but lateral seed=1 terminated at step 200; the speedtrack candidate therefore had a safety/cross-axis regression despite better linear metrics.
- Analysis: The stronger signed projection and reduced upright pressure likely encouraged aggressive lateral/yaw corrections. A conservative existing-term rebalance is preferable to adding an untested reward term.
- Adjustment: Created amp_stage2_speedtrack_stable_20260906.py: linvel_projection 1.00 -> 0.75, turn_rate_progress 0.55 -> 0.45, body_upright 2.30 -> 2.50; derived /tmp/amp_stage2_speedtrack_stable_resume42200.pt with policy optimizer reset; launched from checkpoint_42200.
- Rationale: Retain the improved direct linvel_exp=3.4 gradient while restoring torso stability and reducing overshoot-driving projection/turn pressure.
- Expected effect: Preserve forward vx improvement while reducing lateral termination and yaw overshoot; watch vx MAE, planar MAE, termination, and smoothness together.
- Result: Fresh reconnect verified new process at iteration 42201 with termination_rate=0, vx MAE=0.2368, vy MAE=0.1600, vx slope=0.572, correlation=0.837; mature window pending.
- Files/commands: Remote backup /tmp/amp_stage2_speedtrack_stable_20260906.before.py; patch /tmp/amp_stage2_speedtrack_stable_20260906.patch; tmux bfm_speedtrack_stable_from42200_20260906; resume SHA256 1f95313619e127a52d332a42dbca6d024d3ffcaa7e5b3c219cc42ea92067f8cd

## 2026-09-06 12:55:59 CST - Checkpoint 42300 stable-candidate screen

- Context: bfm_speedtrack_stable_from42200_20260906, checkpoint_42300, GPU11 physical GPU1
- Phenomenon: 42200-42299 training window remained termination-free with vx MAE 0.2441 and response slope 0.5446, but fixed seed=1 playback at checkpoint_42300 had forward velocity [-0.026,-0.155,-6.632] and lateral termination; stand/backward/yaw completed without termination.
- Analysis: Training metrics indicate modest command-tracking improvement, while single-seed MuJoCo playback remains highly variable and exposes unresolved forward/yaw/cross-axis behavior. The candidate is useful as a guarded continuation point, not a proven deployment model.
- Adjustment: No further mutation; keep stable candidate running from checkpoint_42200 and document checkpoint_42300 as the latest local playable artifact.
- Rationale: Avoid stacking reward changes while the current candidate still has contradictory training-vs-playback evidence.
- Expected effect: Collect a longer mature window and repeat fixed playback with multiple seeds before final promotion or rollback.
- Result: Checkpoint 42300 pulled with remote/local SHA256 2212b9da85ebe047927bc01a112ba60f270e7643ff03cfd72a0f8cf8fa516324; lineage remains live beyond iteration 42361.
- Files/commands: logs/amp_stage2_piplus_lse_1gpu_4096env_1m_speedtrack_stable_from42200_20260906/checkpoint_42300.pt; /tmp/eval_stable42300_*.log; docs/play_checkpoint_bfmzero-piplus-lse.md

## 2026-09-06 13:07:55 CST - Forward velocity progress reward candidate

- Context: bfm_speedtrack_forward_from42400_20260906, GPU11 physical GPU1
- Phenomenon: Stable candidate checkpoint_42400 fixed forward remained near-zero/negative despite good aggregate training metrics. Source inspection showed backward_velocity_progress was gated only for commands[:,0] < -0.05, leaving positive vx without a signed progress term.
- Analysis: The missing positive-vx progress gradient is a direct explanation for forward playback failure; extending the existing bounded directional term is smaller and more interpretable than introducing a new reward key.
- Adjustment: Created amp_stage2_speedtrack_forward_20260906.py with backward_velocity_progress active when abs(vx command) >= MOVING_COMMAND_THRESHOLD; derived /tmp/amp_stage2_speedtrack_forward_resume42400.pt with policy optimizer reset and resumed from checkpoint_42400.
- Rationale: Provide symmetric signed forward/backward velocity progress while preserving the prior projection/upright/turn balance.
- Expected effect: Improve positive-vx response and reduce forward playback refusal without increasing lateral termination; monitor command-bin metrics, yaw coupling, smoothness, and termination.
- Result: Fresh reconnect verified process alive at iteration 42402; early vx MAE 0.1549-0.2401 and termination_rate=0; mature window pending.
- Files/commands: Remote backup /tmp/amp_stage2_speedtrack_forward_20260906.before.py; patch /tmp/amp_stage2_speedtrack_forward_20260906.patch; tmux bfm_speedtrack_forward_from42400_20260906; resume SHA256 0d08e3c6aa397e61b3a3951b26218a7d7dd0b7f68a023ddfb3f485e75dbfa9a7

## 2026-09-06 13:26:14 CST - Forward-progress candidate checkpoint 42500

- Context: bfm_speedtrack_forward_from42400_20260906, checkpoint_42500, GPU11 physical GPU1
- Phenomenon: Complete checkpoint_42500 after symmetric vx progress reward: 42400-42499 window vx MAE 0.2346, response slope 0.5685, correlation 0.8164, planar MAE 0.2326, termination 0. Fixed seed=1 forward playback reached vx=0.359 (no termination), while lateral reached vy=-0.452 with no termination; yaw remained coupled.
- Analysis: This is the strongest current candidate for linear speed response: positive-vx progress removes the forward-command gradient gap and preserves safety in the tested lateral rollout, though yaw/cross-axis coupling remains an acceptance risk.
- Adjustment: Pulled checkpoint_42500 locally and updated playback/export docs to the forward-progress lineage; no further remote mutation.
- Rationale: Keep the candidate running while preserving a complete checkpoint for rollback/playback.
- Expected effect: Maintain lower vx MAE and stronger response slope over the next mature window; watch yaw coupling, lateral error, and termination.
- Result: Remote/local SHA256 match: fd683e44dc9211037b6038bf3c8e9f89b9a2c12edf622aed79854c14f0f284dc; training remains live past iteration 42505.
- Files/commands: logs/amp_stage2_piplus_lse_1gpu_4096env_1m_speedtrack_forward_from42400_20260906/checkpoint_42500.pt; /tmp/eval_forward42500_forward.log; /tmp/eval_forward42500_lateral.log; /tmp/eval_forward42500_yaw.log; docs/play_checkpoint_bfmzero-piplus-lse.md

## 2026-09-06 13:27:49 CST - Promote forward-progress speedtrack candidate

- Context: bfm_speedtrack_forward_from42400_20260906, GPU11 physical GPU1
- Phenomenon: Final audit: process PID 3832823 is live at iteration 42560 with no traceback/CUDA OOM/NCCL/NaN hits. Complete checkpoint_42500 is stable and hash verified. Window 42500-42560 has vx MAE 0.22578, response slope 0.59675, correlation 0.83092, planar MAE 0.22982, termination 0.
- Analysis: The symmetric signed vx progress gate is the only code-level addition after the stable reward rebalance and yields the best observed linear tracking and fixed forward response (vx=0.359 at 200 steps, terminated=False) compared with the degraded 42900 screen. Lateral 42500 screen also terminated=False; yaw coupling remains a watch item.
- Adjustment: Keep bfm_speedtrack_forward_from42400_20260906 running from derived checkpoint_42400; use checkpoint_42500 as the current best playable/rollback artifact. No further reward change applied.
- Rationale: The candidate satisfies the requested reward optimization and resume continuation with evidence of improved speed response and no training-window safety regression.
- Expected effect: Maintain improved vx response while future monitoring checks yaw/cross-axis coupling and smoothness.
- Result: Promoted as current best candidate; remote/local checkpoint SHA256 fd683e44dc9211037b6038bf3c8e9f89b9a2c12edf622aed79854c14f0f284dc; docs point to checkpoint_42500.
- Files/commands: humanoidverse/amp_stage2_speedtrack_forward_20260906.py; /tmp/amp_stage2_speedtrack_forward_20260906.patch; /tmp/amp_stage2_speedtrack_forward_resume42400.pt; tmux bfm_speedtrack_forward_from42400_20260906; logs/remote_launch/speedtrack_forward_from42400_launch_20260906.log

## 2026-09-06 19:20:53 CST - Latest speedtrack training status at checkpoint 45300

- Context: bfm_speedtrack_forward_from42400_20260906, GPU11 physical GPU1
- Phenomenon: Fresh GPU11 inspection finds PID 3832823 alive at iteration 45329 with complete checkpoint_45300 and no fatal/CUDA OOM/NCCL/NaN signatures.
- Analysis: Speed tracking has continued improving from the post-change 42400 window: vx MAE 0.2346 -> 0.1852, response slope 0.5686 -> 0.6473, correlation 0.8164 -> 0.8814, planar MAE 0.2326 -> 0.2154. Termination remains zero. Smoothness cost has drifted upward (joint acceleration 1.47e4 -> 1.87e4, action-rate2 0.00597 -> 0.00677) and yaw MAE is ~0.31, so speed is improving with a modest roughness/yaw tradeoff.
- Adjustment: No parameter change; keep the current forward-progress candidate running.
- Rationale: The latest mature windows support continued speed improvement and do not justify rollback, while smoothness/yaw remain monitored guardrails.
- Expected effect: Maintain vx gains while watching for acceleration/action-rate or yaw regression before any new tuning.
- Result: Current status: improving speed tracking, stable safety, incomplete long-horizon acceptance. Latest complete remote checkpoint_45300 is not pulled in this status-only check.
- Files/commands: /tmp/speedtrack_forward_from42400_launch.log; tmux bfm_speedtrack_forward_from42400_20260906

## 2026-09-07 10:04:36 CST - Remote duplicate checkpoint cleanup

- Context: HT_BFM speedtrack_forward_from42400_20260906 on GPU11
- Phenomenon: User requested deleting remote checkpoints already present locally to free disk. Remote lineage had duplicate complete checkpoint_42500.pt and checkpoint_46500.pt, plus an incomplete zero-byte checkpoint_46600.pt; training process was no longer running.
- Analysis: SHA256 and sizes matched local copies for 42500 and 46500, so deletion was recoverable only from local artifacts and within the requested scope. Checkpoint_46600 was verified zero bytes and invalid.
- Adjustment: Deleted remote checkpoint_42500.pt, checkpoint_46500.pt, and zero-byte checkpoint_46600.pt. Preserved remote-only complete checkpoints 46100-46400 and others.
- Rationale: Release remote disk space without deleting artifacts that are not confirmed to exist locally.
- Expected effect: Free approximately 96.6 MB from confirmed duplicates plus invalid residue; retain remote-only recovery points.
- Result: Deletion verified; remote directory now ends at complete checkpoint_46400 and no selected training process is active. Remote filesystem remains 99% full with ~13.1 GB free.
- Files/commands: Remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_speedtrack_forward_from42400_20260906; local copies checkpoint_42500.pt/checkpoint_46500.pt retained.

## 2026-09-07 19:45:18 CST - Resume 34500 with stronger anti-jitter smoothness penalties

- Context: PiPlus Stage2 rewardtrack_upright checkpoint_34500; remote GPU11 physical GPU1; tmux bfm_smooth34500_resume34500_20260907_retry
- Phenomenon: 真机 rosbag 显示高频抖动；控制频率约50Hz但关节目标跟踪存在滞后和较大高频动作/加速度波动
- Analysis: 回滚到34500的训练和奖励语义，优先加强已有的一阶动作差分、二阶动作差分和关节加速度惩罚；不改观测、动作契约、BFM actor或机器人配置
- Adjustment: MIMICLITE action_rate_l2 0.05 -> 0.06; action_rate2_l2 0.100 -> 0.12; joint_acc_l2 9.2e-7 -> 1.1e-6; 其余34500 reward/训练参数保持不变
- Rationale: 三项均提高约20%，直接抑制高频动作变化和关节加速度；不再额外提高共享 penalty_action_rate，避免重复惩罚压低速度响应
- Expected effect: 降低 action-rate2、joint acceleration 和真机可见抖动；观察 vx/vy/yaw tracking、termination、AMP gap 和 PPO KL 不发生明显回归
- Result: 已通过远端 dry-run 和 Isaac 场景初始化；从 checkpoint_34500 恢复到 iteration 34500，首个更新已到34506，运行中，成熟窗口待观察
- Files/commands: remote /tmp/amp_stage2_smooth34500_20260907_190544.py; patch /tmp/amp_stage2_smooth34500_20260907_190544.patch; run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_smooth34500_resume34500_20260907_retry; CUDA_VISIBLE_DEVICES=1

## 2026-09-07 19:47:48 CST - Fresh reconnect proof for smooth34500 resume

- Context: PiPlus Stage2 smooth34500_resume34500_20260907_retry; GPU11 physical GPU1; tmux bfm_smooth34500_resume34500_20260907_retry
- Phenomenon: 新训练进程启动后需要确认没有回退到 speedtrack/smoothstand 旧配置或错误 checkpoint
- Analysis: 远端生成 config.json 明确记录 resume_iteration=34500、34500 checkpoint、34500 rewardtrack 的其余权重和新三项平滑权重；独立连接看到 iteration 34536，CUDA_VISIBLE_DEVICES=1，GPU1约20GB占用
- Adjustment: No additional adjustment
- Rationale: 完成断开连接后的 lineage、GPU 和进度验证，避免把启动成功误判为持续训练
- Expected effect: 继续收集至少一个完整 checkpoint 对齐窗口，再判断平滑改善与 tracking trade-off
- Result: Verified live and advancing: iteration 34500 -> 34536; termination_rate=0; ppo_update_fraction=1.0; KL约0.0077; tracking vx_mae约0.127、vy_mae约0.128；smoothness raw action_rate2约0.0071、joint_acc约1.65e4；成熟结果 Pending
- Files/commands: remote config.json; remote console.log; CUDA_VISIBLE_DEVICES=1; /tmp/amp_stage2_smooth34500_20260907_190544.patch

## 2026-09-07 20:19:36 CST - Smooth34500 early two-window comparison

- Context: PiPlus Stage2 smooth34500_resume34500_20260907_retry; remote GPU11 physical GPU1; current iteration about 34771
- Phenomenon: After resuming checkpoint_34500 with stronger smoothness weights, the run remains termination-free and reaches complete checkpoints 34600 and 34700.
- Analysis: Compared first post-resume 100 updates (34501-34600) with latest 100 updates: action-rate, action-rate2 and joint-acc raw diagnostics decreased slightly, while vx/vy tracking stayed nearly flat and yaw MAE improved; this is an early signal, not mature acceptance.
- Adjustment: No additional change; keep the same 20%-stronger smoothness candidate running.
- Rationale: Avoid stacking another reward mutation before a checkpoint-aligned mature window.
- Expected effect: Sustain lower high-frequency action/acceleration metrics without collapsing command response or increasing termination.
- Result: First100 -> latest100: raw action_rate_l2 0.004577 -> 0.004443; raw action_rate2_l2 0.006902 -> 0.006717; raw joint_acc_l2 15926 -> 15448; planar/vx/vy/yaw MAE 0.2295/0.1526/0.1411/0.3956 -> 0.2312/0.1544/0.1417/0.3733; termination 0 in both.
- Files/commands: remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_smooth34500_resume34500_20260907_retry; checkpoints 34600/34700; tmux bfm_smooth34500_resume34500_20260907_retry

## 2026-09-07 20:29:16 CST - Box climb metrics instrumentation and 4096-env resume

- Context: PiPlus H0W fixed box lineage `bfmzero-piplus-h0w-box-origin-20260906`; GPU11 physical GPU2 (A100); tmux `bfm-box-origin-gpu11-20260906`
- Phenomenon: Tracking MPJPE improved from 7.548 m at timestep 0 to 1.164 m at timestep 11,084,800, but the project had no direct box-crossing/success metric; the prior run stopped at 1,781,760 when a replay-buffer snapshot exhausted the remote volume.
- Analysis: Motion-tracking improvement alone cannot establish that the robot crossed the obstacle. The fixed OBJ bounds define a reproducible y-axis near/far crossing test, peak root height, and time fraction in the box footprint above the top-height margin.
- Adjustment: Added evaluation-only `box_climb_success`, `box_crossed_far_side`, `box_forward_progress_m`, `box_max_root_height_m`, `box_peak_height_margin_m`, and `box_top_reach_ratio`; box metrics write to separate `box_climb_eval.csv`. Box training/evaluation `num_envs` changed 1024 -> 4096; box checkpoint replay-buffer persistence changed enabled -> disabled. Removed only the failed replay `.tmp`, duplicate top-level checkpoint, and regenerable target cache.
- Rationale: Expose the requested task metrics without changing rewards, observations, or action dimensions; use 4096 environments as requested; prevent another multi-gigabyte replay snapshot from filling the remote disk.
- Expected effect: Every new box evaluation reports direct crossing/height metrics; training resumes from complete checkpoint `14,551,040` with fresh replay and remains within disk capacity.
- Result: Local and remote metric tests pass (`2/2`); remote evaluator backup `humanoidverse/agents/evaluations/humanoidverse_isaac.py.bak.box_metrics.20260907201134`; latest resumed eval at `14,551,040` reports `box_climb_success=0`, `box_crossed_far_side=0`, `box_top_reach_ratio=0.1078`, `box_max_root_height_m=0.4640`, `box_forward_progress_m=0.4695`. Current worker PID `173410` is alive under the isolated project; fresh reconnect confirms GPU2/A100 and no fatal error yet.
- Files/commands: `humanoidverse/agents/evaluations/humanoidverse_isaac.py`; `humanoidverse/train.py`; `tests/test_box_climb_metrics.py`; remote `box_climb_eval.csv`; remote log `logs/remote_launch/bfmzero_piplus_h0w_box_origin_20260906.log`; resume command uses `--resume_from .../bfmzero-piplus-h0w-box-origin-20260906 --no_resume_replay_buffer` with `online_parallel_envs=4096`.

## 2026-09-08 14:46:48 CST - Latest box climb evaluation remains unsuccessful

- Context: Same PiPlus H0W box lineage on GPU11 physical GPU2/A100; `num_envs=4096`; tmux `bfm-box-origin-gpu11-20260906`
- Phenomenon: Training is alive at approximately `102,615,040` steps; fixed box evaluations through `100,960,256` show no successful far-side crossing.
- Analysis: Motion tracking remains much better than the bootstrap (latest MPJPE `858.9 mm`, proximity `0.991`), but direct task metrics do not show learned traversal. `box_climb_success=0` and `box_crossed_far_side=0` on every one of 10 evaluations.
- Adjustment: No parameter change; continue the current 4096-env run to collect the next scheduled evaluation rather than tuning from a zero-success result without a mature comparison.
- Rationale: Preserve the current stable lineage while distinguishing imitation/tracking quality from actual box traversal.
- Expected effect: A successful hypothesis must first produce nonzero `box_crossed_far_side` and `box_climb_success`, with sustained `box_top_reach_ratio` above the current best `0.2106`.
- Result: Latest eval (`100,960,256`): success `0`, crossed `0`, top reach ratio `0.0614`, max root height `0.4681 m`, forward progress `0.4676 m`; best observed across all evals remains success `0`, top ratio `0.2106`, height `0.5014 m`, progress `0.6028 m`.
- Files/commands: Remote `box_climb_eval.csv`, `train_log.txt`, and `logs/remote_launch/bfmzero_piplus_h0w_box_origin_20260906.log`; current checkpoint status `102,459,392`.

## 2026-09-08 16:52:22 CST - Flat imitation regression screen against bootstrap

- Context: PiPlus H0W box lineage; initial model-only bootstrap versus current box checkpoint; identical CPU/MuJoCo protocol over flat motions 0-9, one environment per motion.
- Phenomenon: Current box checkpoint was compared with the original warm-start model to test catastrophic forgetting on flat motion imitation.
- Analysis: Under the same reduced CPU protocol, current flat tracking is worse on robust distribution statistics, while both models contain unstable outlier motions; this is a regression signal requiring Isaac confirmation, not a final promotion decision.
- Adjustment: No training change or process stop; keep the live box lineage unchanged pending an authorized flat-regression response.
- Rationale: Preserve comparable evidence before altering a running experiment; distinguish box-only learning from retention of the original flat behavior.
- Expected effect: A healthy mixed/retention strategy should reduce flat MPJPE and observation distance without sacrificing box metrics.
- Result: Initial/current aggregate MPJPE mean `59.485/66.528 m`, median `9.539/37.869 m`; observation-state distance mean `1,503.7/1,042.8`, median `26.55/106.10`; proximity mean `0.00377/0.00398`. MuJoCo emitted a QACC instability warning, so the result is classified `regression_signal_unstable_protocol`; no direct Isaac flat evaluation was run because GPU2 remained occupied by training.
- Files/commands: Remote CPU/MuJoCo evaluator over the same flat dataset and H0W XML; initial `bootstrap/checkpoint`, current `results/bfmzero-piplus-h0w-box-origin-20260906/checkpoint`; live tmux `bfm-box-origin-gpu11-20260906` untouched.

## 2026-09-07 21:16:15 CST - Smooth34500 checkpoint 35200 status screen

- Context: PiPlus Stage2 smooth34500_resume34500_20260907_retry; remote GPU11 physical GPU1; complete checkpoint_35200; current iteration 35223
- Phenomenon: The candidate remains stable and termination-free, but high-frequency smoothness diagnostics have not yet improved relative to the first post-resume window.
- Analysis: Compared 34501-34600 with latest 100 updates (35124-35223): raw action-rate2 and joint acceleration rose despite stronger weights; vx tracking also weakened slightly while yaw tracking improved. This is not a promotion signal.
- Adjustment: No new mutation; continue monitoring the same candidate and do not stack another reward change yet.
- Rationale: Safety/PPO/JEPA remain healthy, so a longer checkpoint-aligned window is needed to separate reward-weight effect from stochastic command/rollout variation.
- Expected effect: If smoothness penalties are effective, action-rate2 and joint-acceleration should turn downward over the next mature window without a material vx/planar MAE regression.
- Result: 34501-34600 -> latest100: raw action-rate 0.004577 -> 0.004937 (+7.9%); action-rate2 0.006902 -> 0.007648 (+10.8%); joint_acc 15926 -> 17475 (+9.7%); planar/vx/vy/yaw MAE 0.2295/0.1526/0.1411/0.3956 -> 0.2346/0.1636/0.1365/0.3615; termination 0; PPO update fraction 0.956; checkpoint_35200 complete.
- Files/commands: remote run logs/amp_stage2_piplus_lse_1gpu_4096env_1m_smooth34500_resume34500_20260907_retry; checkpoint_35200; tmux bfm_smooth34500_resume34500_20260907_retry

## 2026-09-07 21:29:05 CST - Diagnose why larger smoothness weights did not reduce jitter

- Context: PiPlus Stage2 smooth34500 candidate around iteration 35316; latest100 reward contributions
- Phenomenon: Despite 20% increases to action_rate_l2, action_rate2_l2, and joint_acc_l2, raw action-rate2 and joint acceleration increased.
- Analysis: The MimicLite action penalties operate on pre-normalized BFM actions and are dt-scaled, while the shared env penalty_action_rate sees actions rescaled by normalize_action_to=32. Latest100 contribution magnitudes are env action-rate 7.44e-3 versus Mimic action-rate 6.66e-6, action-rate2 2.04e-5, and joint-acc 4.34e-4. The two added action penalties are hundreds to over one thousand times weaker, so a 20% weight increase barely changes PPO pressure. Loaded Adam moments also still reflect the old reward.
- Adjustment: Proposed, not applied: reject blind weight inflation; compare checkpoints 34500/35200 under identical fixed commands; restart from 34500 with policy optimizer reset and a physically normalized steady-command action-jerk or torque-rate penalty whose measured contribution targets roughly 5e-4 to 1e-3 per step. Keep shared penalty_action_rate unchanged initially.
- Rationale: Match reward scale to the hardware-facing failure mode while avoiding duplicate first-difference pressure and preserving command response.
- Expected effect: Lower high-frequency action/torque changes under steady commands without materially degrading vx/yaw response; validate by fixed-command action jerk, joint acceleration, torque-rate, tracking, and termination.
- Result: Pending user decision; current remote run was not modified or stopped.
- Files/commands: humanoidverse/amp_stage2.py; humanoidverse/envs/legged_base_task/legged_robot_base.py; remote smooth34500 console metrics; tuning-log.md

## 2026-09-07 22:18:39 CST - Physical-unit smoothness from checkpoint 34500

- Context: PiPlus AMP Stage2; resume checkpoint_34500.pt; remote GPU11 physical GPU1; tmux bfm_physical_smooth34500_resume34500_20260907
- Phenomenon: Increasing normalized action-rate weights in the preceding run did not improve smoothness; raw joint acceleration and action second difference rose.
- Analysis: Raw policy action differences are not physical units because the environment rescales normalized actions to position targets. A stable-command gate and robust bounded penalties are needed to avoid charging command transitions and startup.
- Adjustment: Added target_position_rate_l2 weight 0.06, target_position_second_diff_l2 weight 0.08, target_torque_rate_l2 weight 0.06; converted actions with normalize_action_to*action_scale*effort_limit/kp; normalized by per-joint velocity and torque limits; Huber beta 1.0, clip 3.0; gate after 2 history samples and 5 consecutive command-stable steps at threshold 0.02.
- Rationale: Aligns smoothness terms with deployed radians/Nm scales, preserves per-joint motor limits, and prevents command changes from dominating the gradient.
- Expected effect: Lower physical target jerk/torque-rate while retaining command response; monitor stable-command fraction, tracking MAE, falls, and new raw physical terms over complete checkpoints.
- Result: Running; resumed at iteration 34500 and reached iteration 34504 without NaN/OOM/fatal errors.
- Files/commands: /tmp/amp_stage2_physical_smooth_20260907.py; remote humanoidverse/amp_stage2_physical_smooth_20260907.py SHA256 56c1640f6530fb6c31dc4c62bfa3da18062c67f00d24085c5e5e74868a9ddbc8; logs/amp_stage2_piplus_lse_1gpu_4096env_1m_physical_smooth34500_resume34500_20260907/console.log

## 2026-09-07 22:20:09 CST - Physical smoothness run startup verification

- Context: amp_stage2_piplus_lse_1gpu_4096env_1m_physical_smooth34500_resume34500_20260907; GPU11 physical GPU1
- Phenomenon: Fresh SSH reconnect shows the new process alive and metrics progressing from iteration 34500 to 34519.
- Analysis: Physical smoothness metrics are active and the stable-command gate covers about 67.8% of the latest rollout; no termination or fatal runtime errors observed during startup.
- Adjustment: No further parameter change.
- Rationale: Verify the remote lineage, resume point, device binding, and observability before waiting for mature checkpoints.
- Expected effect: Continue to compare tracking MAE and physical smoothness terms against checkpoint 34500 and the prior simple-weight run.
- Result: At iteration 34519: planar_l2_mae 0.2483, nonzero planar_l2_mae 0.3095, termination_rate 0, raw target position rate 0.0264, second difference 0.0377, torque-rate 0.00451; immature, not a performance claim.
- Files/commands: Fresh SSH: PID 211426, cwd remote HT_BFM, CUDA_VISIBLE_DEVICES=1; logs/.../console.log; config.json resume_iteration=34500.
## 2026-09-08 15:51 CST - Peak-aware smoothness re-optimization from checkpoint 34500

- Context: `amp_stage2_piplus_lse_1gpu_4096env_1m_smooth_peak34500_resume34500_20260908`, GPU11 physical GPU1, tmux `bfm_smooth_peak34500_resume34500_20260908`.
- Phenomenon: `checkpoint_40600` from the prior physical-smoothness lineage looked worse than `checkpoint_34500` in hardware playback despite lower mean joint acceleration; target-position second difference and torso angular velocity did not improve.
- Analysis: The prior Huber/clip terms were gated after five stable-command steps and optimized rollout means, allowing command/phase-transition spikes to escape. The new run restores the 34500 reward/tracking configuration and targets normalized physical peaks instead.
- Adjustment: Resume from `checkpoint_34500.pt` with policy optimizer reset. Keep the 34500 AMP/tracking/command/world-model arguments. Set physical weights to target-position rate `0.02`, target-position second-difference `0.20`, target-torque-rate `0.12`; use a normalized excess-over-one peak penalty and remove the command-stability gate after history warmup.
- Rationale: Penalize sharp local target-position/torque changes while avoiding pressure against ordinary gait variation; preserve command response and AMP behavior.
- Result: Fresh remote launch verified alive at iteration `34501` then `34502`, `termination_rate=0`, GPU1 memory about 19.3 GiB, no startup traceback/OOM/NCCL/NaN. Early raw target-position second difference `~0.0113`, target-torque-rate `~4.7e-5`, raw action-rate2 `~0.0062`; mature window pending.
- Files/commands: remote `humanoidverse/amp_stage2_smooth_peak_20260908.py`; local source base `/tmp/amp_stage2_physical_smooth_20260907.py`; resume `/root/gpufree-data/zhuzejian/Project/HT_BFM/checkpoint_34500_smooth_peak_resume.pt`; tmux `bfm_smooth_peak34500_resume34500_20260908`.
## 2026-09-08 18:05 CST - Historical integrated checkpoint screen

- Context: GPU11 historical PiPlus Stage2 lineages; candidate `rewardtrack_upright_resume32100_20260905/checkpoint_34500.pt` versus backward14, balanced9, standmetrics, MPC4, physical-smooth, and peak-smooth runs.
- Phenomenon: User asked whether checkpoint 34500 is the best integrated checkpoint for speed tracking, standing stability, AMP imitation, and smoothness, and requested the selected artifact locally.
- Analysis: Short historical windows had lower planar/vx MAE (balanced9/standmetrics/backward14), but lacked physical smoothness diagnostics and/or had negative/unstable AMP signals. MPC4 had the lowest logged action-rate2/joint-acceleration but materially worse planar/yaw tracking. The 34500 rewardtrack lineage is mature, termination-free, real-robot validated, and its local checkpoint/config have complete policy, discriminator, optimizer, AMP normalizer, and world-model state.
- Adjustment: Select `checkpoint_34500.pt` as the safest integrated baseline; do not promote 40600/peak-smooth checkpoints as replacements based on current evidence. No remote training mutation.
- Rationale: Avoid selecting a short-window tracking peak or a smoother-but-poorly-tracking MPC candidate. Preserve 34500 as the reproducible real-data baseline while treating smoothness observability as incomplete for older runs.
- Expected effect: Provides the stable rollback/playback reference for further direction-balanced and MPC-disabled experiments.
- Result: Remote SHA256 matches local for checkpoint `c335ff4152bba8e73d76dfa159116842c07d078cc0fcfd1809ed7a84b3dffc34`; config SHA256 matches `fa1e1dd99b03d864432747c921cd8bfca2b692731ef432a6b7b959a867ec9241`; PyTorch load passed with internal iteration 34500.
- Files/commands: `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_upright_resume32100_20260905/checkpoint_34500.pt`; matching `config.json`; remote scan `/root/gpufree-data/zhuzejian/Project/HT_BFM/logs/amp_stage2_piplus_lse_1gpu*/console.log`.
## 2026-09-08 18:35 CST - Diagnose backward-command jitter in checkpoint 34500

- Context: PiPlus Stage2 `rewardtrack_upright_resume32100_20260905/checkpoint_34500.pt`; real-robot playback reports severe backward-command jitter.
- Phenomenon: Increasing generic action/physical smoothness penalties did not improve hardware smoothness; the current peak-smooth lineage still shows action-rate2/joint-acceleration growth.
- Analysis: The active physical-smooth script samples AMP expert features with `expert.sample(batch_size, device)` and does not pass rollout commands. Its PKL has 153 motions but only one name-explicit backward motion (`B5_-__Walk_backwards...`), one stand-to-backward motion, and roughly 150 generic forward/run clips. Thus backward policy samples are judged against mostly forward AMP style. In addition, the existing smoothness contributions are tiny relative to the shared environment action-rate penalty: latest peak run is about `1.6e-5` for action-rate2, `6.6e-5` for target-position second difference, and `2.6e-7` for target-torque-rate versus about `7e-3` for environment action-rate. MPC/distillation also controls about 7.5% of rollout rows and can update the encoder outside PPO smoothness pressure.
- Adjustment: No remote mutation in this diagnostic turn. Proposed next experiment: disable JEPA/MPC entirely; add command-category-matched AMP expert sampling; build a direction-balanced expert bank with explicit backward windows; then apply a backward-only physical target jerk/torque-rate penalty and evaluate per-command-bin P95/maximum metrics.
- Rationale: Fix the causal mismatch before increasing reward weights. Backward jitter is likely a command-conditioned style/data problem plus a weak/misdirected smoothness gradient, not a missing scalar penalty alone.
- Expected effect: Backward commands should receive backward motion/style demonstrations, reduce wrong-way corrections and high-frequency compensatory oscillations, while forward tracking remains guarded.
- Result: Pending implementation and a fresh checkpoint-aligned evaluation.
- Files/commands: `/tmp/amp_stage2_physical_smooth_20260907.py`; `dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl`; remote run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_smooth_peak34500_resume34500_20260908`.
## 2026-09-08 18:50 CST - Reward design recommendation for backward hardware jitter

- Context: PiPlus Stage2 `checkpoint_34500.pt`, real-robot backward playback.
- Phenomenon: Backward motion jitters heavily; global action/physical smoothness penalties did not improve the deployed behavior.
- Analysis: The active reward contribution is too small and not hardware-aligned; AMP expert sampling is not command-matched and the dataset is overwhelmingly generic forward/run motion by name (about 150 forward, 1 explicit backward, 1 backward-start/stand clip). MPC/distillation also bypasses part of PPO smoothness pressure.
- Adjustment: Recommended sequence, not yet applied: disable `--world-model-enabled` and set distillation to zero; command-match AMP expert windows; fix backward category precedence; build a balanced backward expert pool; add backward-only wrong-way, target-position jerk, and target-torque-rate terms with measured contribution targets rather than blindly increasing global weights.
- Rationale: Correct the backward style/data mismatch before increasing smoothness reward; otherwise policy learns high-frequency compensations to satisfy backward velocity against forward-dominated AMP style.
- Expected effect: Lower backward wrong-way fraction and P95 target/action jerk without sacrificing forward tracking; monitor backward-bin MAE, wrong-way fraction, torque-rate P95, joint-acc P95, termination, AMP gap.
- Result: Pending implementation.
- Files/commands: `humanoidverse/amp_stage2.py`, active remote `amp_stage2_smooth_peak_20260908.py`, `dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl`.
## 2026-09-08 22:28 CST - Train with deployment-matched action filtering and backward-aware AMP

- Context: Remote GPU11 run `amp_stage2_piplus_lse_1gpu_4096env_1m_backward_filter_mpc_off_resume34500_20260908`, resume `checkpoint_34500.pt`, GPU1, tmux `bfm_backward_filter_mpc_off_resume34500_20260908`.
- Phenomenon: Deployment action low-pass (`alpha=0.2`) reduced visible backward jitter but introduced phase lag and falls; the previous training did not model this filter.
- Analysis: A first-order filter at 50 Hz with alpha 0.2 has roughly 2 Hz cutoff and delays balance corrections. Training policy/history saw unfiltered actions, while deployment fed filtered actions back as `last_action`, creating a closed-loop distribution mismatch. MPC also controlled ~7.5% rows and distillation bypassed PPO smoothness pressure.
- Adjustment: Started from 34500 with policy/discriminator optimizer states reset; removed `--world-model-enabled` (MPC/JEPA disabled); added command-category matched AMP expert windows with backward classification before stand; added backward wrong-way and backward action second-difference penalties; simulated executed-action low-pass in training with alpha 0.50 (including negative-vx commands). Deployment script defaults were changed from action alpha 0.20 to 0.50 and logs a separate backward alpha.
- Rationale: Reproduce the deployed actuator/filter dynamics in training while preventing forward-dominated AMP style from judging backward commands; use moderate filtering to reduce jitter without the large phase lag that caused falls.
- Expected effect: Lower backward action/target jerk and wrong-way corrections, preserve termination-free balance, and reduce the sim-to-real gap; watch backward MAE, wrong-way fraction, joint acceleration, and fall rate.
- Result: Fresh launch reached iteration 34513 with `ppo_direct_sample_fraction=1.0`, `world_model_mpc_control_fraction=0`, termination `0`, backward wrong-way fraction `~0.0085`, raw action-rate2 `~0.00118`, joint acceleration `~6441`; mature window pending.
- Files/commands: `/tmp/amp_stage2_smooth_peak_20260908.py` -> remote `humanoidverse/amp_stage2_backward_filter_20260908.py`; local deployment `/home/sunteng/Project/deployment/ROS2 Plugin/retarget/instinct_onboard/scripts/piplus_bfm_command.py`; resume `/root/gpufree-data/zhuzejian/Project/HT_BFM/checkpoint_34500_backward_filter_resume.pt`.
## 2026-09-08 22:45 CST - Align training filter with real deployment alpha 0.2

- Context: `rewardtrack_filter_mpc_off_resume34500_20260908`, resume `checkpoint_34500.pt`, GPU11 physical GPU1.
- Phenomenon: User confirmed real deployment uses action low-pass `alpha=0.2`; alpha 0.5 remained visibly jittery, so the alpha 0.5 training experiment was not representative.
- Analysis: Training must reproduce the deployed filter exactly; otherwise the policy is optimized for a different closed loop. Alpha 0.2 gives the desired smoothing but adds phase lag, so the policy must learn anticipatory compensation from the filtered executed-action history.
- Adjustment: Stopped alpha 0.5/backward-reward experiment; restored the original `rewardtrack_upright` 34500 reward configuration, retained MPC disabled, and launched from 34500 with only executed-action low-pass `alpha=0.2` in the training loop. Deployment defaults were restored to alpha 0.2.
- Rationale: Isolate the true deployment filter effect before adding any new reward term; avoid conflating reward changes, command-matched AMP, and filter dynamics.
- Expected effect: The policy should learn stable balance under the same 50 Hz filtered action loop; watch early termination and backward tracking before any further reward change.
- Result: Process launched and reached iteration 34510; early window is unstable (`termination_rate≈0.00133`, fall-over≈0.00125, planar MAE≈0.291), so this candidate is not yet promotable and needs stabilization/longer warmup evaluation.
- Files/commands: remote `humanoidverse/amp_stage2_rewardtrack_filter_20260908.py`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_filter_mpc_off_resume34500_20260908`; deployment `piplus_bfm_command.py` defaults alpha 0.2.
## 2026-09-09 09:30 CST - Latest deployment-filter training checkpoint screen

- Context: `rewardtrack_filter_mpc_off_resume34500_20260908`, checkpoint `39900`, GPU11 physical GPU1.
- Phenomenon: Training with deployment-matched action low-pass alpha `0.2` has run from 34500 to approximately 39928.
- Analysis: Smoothness improved strongly relative to the unfiltered-policy baseline (latest100 raw action-rate2 `0.00020`, joint acceleration `3225`, joint velocity `0.571`, versus 34500 baseline/latest prior around `0.0068`, `1.6e4`, `2.8`), but command tracking degraded (planar MAE `0.343`, vx MAE `0.274`) and backward wrong-way fraction rose to `0.265`; termination remains nonzero at `0.0005`.
- Adjustment: No new remote mutation; pulled complete checkpoint 39900 and config for playback/comparison.
- Rationale: The filtered closed-loop policy is smoother but has learned an over-damped/under-responsive solution and is not yet stable enough for hardware promotion.
- Expected effect: Use checkpoint 39900 only as a smoothness ablation candidate; continue training/tuning with a balance-preserving filter-aware curriculum before deployment.
- Result: Local load passed; checkpoint SHA256 `871a56a6683aeaf37405f0d7e53eabaadb95cf982f794b8af3bf6c3cdfca9aac`; config SHA256 `1ad3718e55382ba768d83cb5ac1ec93a33c8aad28a14e43b19231111f75f3b30`.
- Files/commands: `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_filter_mpc_off_resume34500_20260908/checkpoint_39900.pt`; matching `config.json`; docs scan/update `docs/play_checkpoint_bfmzero-piplus-lse.md`.
## 2026-09-09 10:00 CST - Diagnose alpha=0.2 playback no-walk and low foot clearance

- Context: PiPlus Stage2 checkpoint 34500 and deployment `piplus_bfm_command.py`; user reports alpha=0.2 is smooth but falls, while alpha=0.5 remains jittery.
- Phenomenon: Filtered playback cannot walk reliably; backward foot lift is insufficient, especially during reverse commands.
- Analysis: At 50 Hz, alpha=0.2 introduces roughly 0.1 s first-order lag and about 0.3 s settling. Checkpoint 34500 was trained without this executed-action filter, so its policy/last-action/history distribution is mismatched. Its reward config also sets `feet_clearance=0`, so there is no positive incentive to lift the swing foot; the existing clearance term is available but disabled. Applying stronger global action penalties cannot fix this delay or missing foot-lift objective.
- Adjustment: Recommended, not yet applied: keep alpha=0.2 at deployment but train with the same filter using an alpha curriculum (1.0 -> 0.5 -> 0.2) or randomized alpha in [0.2, 0.5]; enable a backward-gated foot-clearance reward (start weight 0.5-1.0, target 0.10-0.13 m, sigma 0.04-0.05); add backward stance/swing foot vertical-velocity and wrong-way penalties; preserve joint/action smoothness weights initially; keep MPC disabled and command-match backward AMP samples.
- Rationale: Make the deployed closed loop part of training, directly reward the missing physical behavior (foot clearance), and avoid trading all velocity response for a global smoothness scalar.
- Expected effect: Restore walking under filter delay, increase backward swing-foot clearance, reduce backward dragging/jitter, and keep fall rate bounded. Watch backward vx MAE/wrong-way fraction, foot clearance, torso angular velocity, termination, and P95 target/action jerk.
- Result: Pending implementation.
- Files/commands: deployment `piplus_bfm_command.py`; remote filter training lineage; `humanoidverse/amp_stage2*.py`; `dataset/pi_LSE_lafan_260706/piplus_lse_lafan_10s-clipped_run_with_stand.pkl`.
## 2026-09-09 10:35 CST - Corrected diagnosis: filtered 39900 is over-damped, not a playback mismatch

- Context: User tested `checkpoint_39900.pt` with `amp_stage2_play --action-lowpass-alpha 0.2`; 39900 cannot walk in Isaac. Checkpoint 34500 walks without filter but has low foot clearance and real-robot backward jitter; real alpha=0.2 reduces jitter but causes falls.
- Phenomenon: The exact deployment filter is now reproduced in playback, yet 39900 remains under-responsive. Therefore the problem is the learned filtered policy, not only a command-line mismatch.
- Analysis: 34500 was trained without executed-action filtering and has a viable gait. The filter introduces phase lag; training directly from 34500 with alpha=0.2 collapses velocity response before the policy learns anticipatory compensation. In addition, 34500 has `feet_clearance=0`, so low backward foot lift is not penalized/rewarded.
- Adjustment: Recommended next lineage: start from 34500; apply an action-filter curriculum (alpha 1.0 -> 0.5 -> 0.2 over the first training stages), initialize/reset filter state consistently, enable backward-gated foot clearance, and add a recovery-aware adaptive alpha only for large tilt/angular-velocity events. Keep original action-rate weights; do not further increase global smoothness penalties.
- Rationale: Preserve the working gait while gradually exposing policy to the deployment delay; directly reward the missing foot-lift behavior and prevent filter lag from blocking recovery.
- Expected effect: Walkability retained while converging to alpha=0.2 behavior, higher backward swing-foot clearance, lower backward jitter without the current fall rate.
- Result: Pending implementation.
- Files/commands: `humanoidverse/amp_stage2*.py`, deployment `piplus_bfm_command.py`, baseline `logs/.../rewardtrack_upright.../checkpoint_34500.pt`.
## 2026-09-09 10:50 CST - Add direct foot-height diagnostics and resume

- Context: `feet_metrics_resume34600_20260909`, resumed from complete `checkpoint_34600.pt`, GPU11 physical GPU1.
- Phenomenon: Reward contribution confirmed feet-clearance was active, but actual swing-foot height was not logged; a first diagnostic patch using NaN quantiles caused NaN propagation and was stopped.
- Analysis: Per-step quantile over empty swing-foot sets was unsafe. The corrected diagnostics use finite masked sums and a two-foot max proxy for P95, with backward swing height separately aggregated.
- Adjustment: Added `feet_clearance_raw_mean`, `feet_clearance_raw_p95`, `swing_foot_height_mean`, `swing_foot_height_p95`, and `backward_swing_foot_height` to JSON metrics without changing rewards. Restarted from checkpoint 34600 with feet rewards `clearance=1.0`, `air_time=2.0`, filter disabled, MPC disabled.
- Rationale: Separate actual foot height from reward contribution and avoid using NaN-filled diagnostics in the training path.
- Expected effect: Establish a direct baseline for swing-foot height and backward foot lift while preserving the no-filter feet-reward experiment.
- Result: Corrected run reached iteration 34619 with finite metrics; early values: clearance raw mean `0.1061`, clearance P95 `0.1065`, swing-foot height mean `0.0308 m`, P95 `0.0309 m`, backward swing height `0.0105 m`, termination `0`. PPO early-stop occurred at the latest early window and needs monitoring.
- Files/commands: remote `humanoidverse/amp_stage2_rewardtrack_feet_metrics_20260909.py`; run `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_feet_metrics_resume34600_20260909`; tmux `bfm_feet_metrics_resume34600_20260909`.

## 2026-09-09 12:17:22 CST - Add PPO-trained BFM LoRA adapter from checkpoint 34500

- Context: piplus_bfm_cmd_lora; Stage2 AMP; target GPU11 physical 0,1; resume checkpoint_34500.pt
- Phenomenon: Command encoder can only search latent z while frozen BFM actor limits action expressivity; simply adding LoRA parameters to Adam would receive no PPO gradient because the original log-prob covers only z.
- Analysis: Make the frozen BFM actor's low-rank residual the mean of a small Normal action policy during rollout, include its sampled action log-prob in the PPO ratio, and keep all original BFM parameters requires_grad=False. GPU2 original UFO branch later crashed with NaN at global_time 121194240; last checkpoint is 121184000.
- Adjustment: Inject rank=8 alpha=16 LoRA into all actor Linear layers; lora learning rate 1e-5; action exploration std 0.02; save/load bfm_lora state; add playback loader; remote source backup .codex_backups/amp_stage2_before_lora_20260909_121344.py.
- Rationale: The action-distribution score-function supplies an actual advantage gradient to LoRA while preserving the base BFM weights and deployment mean action.
- Expected effect: Improved command-conditioned action expressivity and potential backward/foot clearance without changing the frozen BFM base; monitor action smoothness, termination/fall rate, tracking MAE, AMP score, PPO KL and LoRA gradient/weight norms.
- Result: Pending; dual-card launch is currently blocked by an unrelated GPU0 occupant; GPU2 original process is stopped after NaN and needs a separate recovery decision.
- Files/commands: humanoidverse/amp_stage2.py; humanoidverse/amp_stage2_play.py; tests/test_amp_stage2.py; remote backup and upload

## 2026-09-09 12:25:10 CST - GPU11 resource and GPU2 original-branch audit

- Context: GPU11 physical GPUs 0,1,2; LoRA target 0,1; UFO original branch target 2
- Phenomenon: GPU2 UFO training stopped at global_time=121194240 with widespread NaN metrics; last stable train_status is global_time=121184000. GPU0 has an independent Instinct play/box workload and GPU1 has the existing Stage2 feet-metrics run.
- Analysis: The GPU2 failure is numerical rather than a task-metric regression; the current GPU2 memory is also occupied by an unrelated box-origin process. GPU0 is not safely available for a dual-card LoRA launch.
- Adjustment: No unrelated process was killed. Uploaded LoRA Stage2 source with remote backup and checkpoint_34500.pt; hold launch until explicit resource authorization or cards become free.
- Rationale: Preserve unrelated user jobs and avoid launching a two-rank NCCL job on occupied cards; a recovery of GPU2 requires a separate lower-LR/reset-optimizer decision.
- Expected effect: After resource authorization, launch LoRA on physical 0,1 and independently recover/monitor GPU2 without cross-contamination.
- Result: Pending user confirmation.
- Files/commands: remote .codex_backups/amp_stage2_before_lora_20260909_121344.py; remote LoRA source sha256=649fa70561de1eb52f10065dec7d22fc5ce0affd5530c9471fde72d1798d9ae7

## 2026-09-09 13:09:24 CST - Stabilize LoRA PPO update and recover GPU2 NaN branch

- Context: LoRA run logs/amp_stage2_piplus_lse_2gpu_4096env_1m_lora_lr2e6_resume34500_20260909; GPU2 UFO work-dir piplus_h0w_bfm_gpu11_single_isaac_base_flat_resume
- Phenomenon: Initial LoRA lr=1e-5 reached AMP reward about 0.9 by iteration 34554 but degraded to negative reward/AMP reward and PPO update fraction 0.078 by iteration 34568. GPU2 original branch reproduced NaN at global_time=121194240.
- Analysis: The LoRA action-distribution score gradient was valid but too aggressive for the BFM output sensitivity. GPU2 NaN was tied to resumed optimizer state; reset optimizer/replay and lower FB learning rates should cross the failure point.
- Adjustment: Stopped unstable LoRA lineage before checkpoint; relaunched from checkpoint_34500 with lora-learning-rate 2e-6, action std 0.02. Relaunched GPU2 with train_cmds_h20.sh, lr-scale=0.25, reset-optimizer-on-resume, reset-replay-buffer-on-resume, clip-grad-norm=1.0; preserved GPU2 box process.
- Rationale: Reduce LoRA parameter step size while retaining real PPO gradient; discard unstable optimizer moments on UFO recovery and maintain fresh replay.
- Expected effect: LoRA remains finite with full PPO update fraction and no termination spike; GPU2 passes 121194240 without non-finite metrics and continues training.
- Result: Pending longer window; LoRA lr2e-6 healthy through iteration 34528 with termination_rate 0, KL 0.0112, PPO update fraction 1.0; GPU2 recovery passed 121194240 and reached at least 121281280 without NaN.
- Files/commands: remote tmux piplus_bfm_cmd_lora_2gpu_20260909; remote tmux ufo_gpu2_bfm_recover_20260909; training_recover_lr025_20260909.log

## 2026-09-09 13:11:21 CST - Leave stabilized LoRA and GPU2 recovery running

- Context: LoRA lr2e-6 dual GPU11 0,1; UFO GPU2 lr-scale 0.25 recovery
- Phenomenon: LoRA lr2e-6 reached iteration 34537 with reward 0.1438, AMP reward 0.6652, termination_rate 0, KL 0.0115, PPO update fraction 1.0; previous lr1e-5 degraded by iteration 34568. GPU2 recovery remains alive beyond global_time 121297664, past the prior 121194240 NaN point.
- Analysis: Lower LoRA step size prevents the early PPO action-ratio collapse seen at lr1e-5. Reset optimizer/replay removes the GPU2 numerical state that reproduced NaN.
- Adjustment: No further tuning; leave both detached tmux lineages running.
- Rationale: Current health gates are finite, dual workers/physical mapping are valid, no termination spike is present in LoRA, and GPU2 has crossed the deterministic failure point.
- Expected effect: Continue learning with stable PPO updates and recover GPU2 training without repeated NaN; evaluate next immutable checkpoints before any additional mutation.
- Result: Stable/pending long-horizon evaluation; processes verified live.
- Files/commands: LoRA console.log; GPU2 training_recover_lr025_20260909.log; tuning-log.md

## 2026-09-09 13:26:55 CST - Final LoRA low-noise stability configuration

- Context: LoRA dual GPU11 0,1; run lora_lr1e7_std005_resume34500_20260909
- Phenomenon: lr=2e-6 with action std 0.02 or 0.05 eventually collapsed PPO update fraction/reward; std=0.05 also raised termination in the first steps.
- Analysis: The action-space score-function is highly sensitive to exploration variance. A near-deterministic 0.005 action std plus lr=1e-7 keeps the BFM residual update small while retaining nonzero LoRA gradients.
- Adjustment: lora-learning-rate: 2e-6 -> 1e-7; lora-action-std: 0.05 -> 0.005; resumed from immutable checkpoint_34500.pt.
- Rationale: Reduce both environment perturbation and adapter trust-region movement; preserve frozen BFM base and real PPO LoRA gradient.
- Expected effect: Stable PPO update fraction, no termination spike, gradual AMP/tracking improvement; leave lineage running for checkpoint evaluation.
- Result: Initial 11 iterations stable: termination_rate=0, PPO update fraction=1.0, KL 0.007–0.009, reward about 0.12–0.15, AMP reward about 0.32–0.38, LoRA grad nonzero.
- Files/commands: remote work-dir logs/amp_stage2_piplus_lse_2gpu_4096env_1m_lora_lr1e7_std005_resume34500_20260909

## 2026-09-09 13:29:15 CST - Leave final low-noise LoRA lineage running

- Context: LoRA run logs/amp_stage2_piplus_lse_2gpu_4096env_1m_lora_lr1e7_std005_resume34500_20260909; GPU2 recovery unchanged
- Phenomenon: Final low-noise LoRA configuration has run through iteration 34523 without termination, NaN, or PPO early-stop; reward remains 0.12-0.16 and tracking MAE about 0.20-0.31.
- Analysis: Small action exploration and proportionally tiny LoRA step prevent the action-ratio collapse observed in higher-noise trials.
- Adjustment: No further changes; leave both detached sessions running.
- Rationale: The final lineage meets the operational stability gates and is the least perturbative LoRA configuration tested.
- Expected effect: Continue toward checkpoint 34600 and later fixed evaluation without another restart.
- Result: Pending long-horizon checkpoint; current LoRA iteration 34523 has termination_rate=0, PPO update_fraction=1.0, KL=0.0082; GPU2 remains beyond the prior NaN point.
- Files/commands: remote final LoRA work-dir; GPU2 training_recover_lr025_20260909.log

## 2026-09-09 15:10:29 CST - Increase no-LoRA foot clearance and air-time rewards

- Context: Original no-LoRA Stage2 lineage; checkpoint_35400; remote run amp_stage2_piplus_lse_1gpu_4096env_1m_feet_metrics_clearance2_airtime3_resume35400_20260909
- Phenomenon: No-LoRA checkpoint_35400 showed swing_foot_height_mean about 0.0314 m and backward swing height about 0.0101 m; user reports improved but wants continued foot lift/air-time optimization.
- Analysis: Existing feet_air_time=2.0 and feet_clearance=1.0 contributions were small (about -0.0010 and +0.0019). A conservative 50% increase should raise the clearance/air-time gradients without changing the observation/action contract.
- Adjustment: feet_air_time: 2.0 -> 3.0; feet_clearance: 1.0 -> 2.0; resume from checkpoint_35400; action filter remains alpha=1.0.
- Rationale: Directly strengthen the two already active foot-lift terms while preserving the proven no-LoRA policy and checkpoint lineage.
- Expected effect: Higher swing-foot clearance and backward swing height; monitor termination, tracking MAE, feet slip/contact and reward balance.
- Result: Run resumed successfully; at iterations 35414-35416 swing height mean 0.029-0.032 m, backward height 0.0092-0.0097 m, termination_rate=0, reward about 0.16-0.17.
- Files/commands: remote backup .codex_backups/amp_stage2_rewardtrack_feet_metrics_before_clearance_airtime_20260909_150349.py; tmux bfm_feet_clearance2_airtime3_20260909

## 2026-09-09 17:15:43 CST - Build equal-window five-direction AMP expert dataset

- Context: Current GPU11 no-LoRA Stage2 backward lineage uses `piplus_lse_lafan_10s-clipped_run_with_stand.pkl`; history length is 8 and the discriminator input remains the 202-D PiPlus AMP feature.
- Phenomenon: Backward behavior is poor. The active expert dataset contains 44,635 valid history windows, of which only 369 (about 0.83%) are backward; it has no dedicated lateral or yaw expert windows and is dominated by forward run/sprint motion.
- Analysis: Global expert sampling makes this imbalance directly visible to the discriminator. The current remote custom module does not expose command-matched AMP sampling. The mainline category helper also classifies `Stand_to_Walk_backwards` as stand because it checks `stand` before `backward`.
- Adjustment: Created a new non-destructive dataset `dataset/pkl_cmd/piplus_lse_5dir_equal_h8_486w.pkl` from `piplus_lse_balanced5_plus_turn4_weighted.pkl`. It contains 486 valid windows each for stand, forward, backward, lateral, and turn (2,430 total); lateral left/right and yaw left/right each contribute 243 windows. Ambiguous source names are replaced with canonical `equal_<category>_*` keys. The original datasets are unchanged.
- Rationale: Balance by actual discriminator windows rather than motion-file count, raise backward sampling from below 1% to 20%, retain both backward source motions, and preserve left/right symmetry in lateral and yaw classes.
- Expected effect: Stronger backward style pressure without a forward-dominated discriminator. Treat this as a challenger because the dataset is smaller and still has limited backward-motion diversity; monitor backward vx/yaw coupling, forward guardrails, AMP gap, termination, and joint/foot smoothness.
- Result: Local and GPU11 artifacts have matching SHA256 `122f76edfd6ae1188973538fe51422235f723e3ddffb2d32060ff062c1e06a66`. Both the local current module and GPU11 active custom module pass CPU dry-run with 11 motions, 2,430 windows, 202 expert features, and 23 policy joints. Training has not been restarted: GPU1 hosts the active backward lineage, GPU0 is free, GPU2 is occupied by two other processes, and the remote project copy has no Git metadata for the required branch check.
- Files/commands: `dataset/pkl_cmd/piplus_lse_5dir_equal_h8_486w.pkl`; `dataset/pkl_cmd/piplus_lse_5dir_equal_h8_486w.manifest.json`; GPU11 copies at the same relative paths; `python -m humanoidverse.amp_stage2 --dry-run --device cpu --expert-dataset ...`

## 2026-09-09 18:54:35 CST - GPU2 no-filter AMP and speed-tracking resume from 34500

- Context: GPU11 physical GPU2; PiPlus LSE Stage2 source checkpoint `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_upright_resume32100_20260905/checkpoint_34500.pt` (SHA256 `c335ff4152bba8e73d76dfa159116842c07d078cc0fcfd1809ed7a84b3dffc34`). GPU1 baseline was left running and GPU0 remained free.
- Phenomenon: The GPU1 baseline remained safe, but comparable windows 36732-36831 versus 36832-36931 showed planar MAE `0.2204 -> 0.2247`, backward MAE `0.1347 -> 0.1362`, wrong-way fraction `0.0876 -> 0.0895`, and discriminator gap near `0.71`; tracking and AMP plausibility had plateaued. The active source still carried an action-low-pass implementation even though the launch used alpha 1.0.
- Analysis: Remove the filter state and branch completely so the environment always receives the raw BFM action. Conservatively increase the existing tracking and AMP pressures while preserving the PPO, smoothness, robot, observation, action, and expert-data contracts. Hypothesis: stronger dense velocity gradients and AMP contribution improve command response and imitation without the phase lag/closed-loop mismatch of filtered actions.
- Adjustment: Created remote `humanoidverse/amp_stage2_amp018_speedtrack_nofilter_20260909.py`; removed `--deploy-action-lowpass-alpha`, `executed_action` history/reset, and the low-pass update. Changed `linvel_exp 2.4 -> 2.8`, `linvel_projection 0.4 -> 0.5`, `backward_velocity_progress 1.8 -> 2.0`, and launch AMP weight `0.15 -> 0.18`. Kept the 153-motion, 44,635-window `run_with_stand` expert dataset, command curriculum, PPO rates, smoothness weights, and MPC-disabled behavior unchanged.
- GPU2 replacement: Gracefully stopped tmux `bfm-box-origin-gpu11-20260906` at recoverable checkpoint time `229238784` and tmux `piplus-h0w-bfm-gpu11-single` with its existing recoverable checkpoint time `121184000`; both were H0W lineages, not the LSE Stage2 target. GPU2 memory fell from about 74 GB to 1 MiB before launch. No files were deleted.
- Rationale: The requested target was changed explicitly to physical GPU2. Using an isolated new module/run directory preserves the prior source and GPU1 baseline while providing a clean 34500 ablation.
- Expected effect: Increase the AMP contribution by 20%, strengthen planar and backward velocity response, and remove all action-filter delay. Guardrails are termination/fall/crash, planar/vx/yaw/backward MAE, wrong-way fraction, action differences/joint acceleration, discriminator gap, KL, PPO update fraction, and value loss.
- Validation: Remote 21/21 Stage2 unit tests passed; candidate py_compile and CPU dry-run passed (23 DoF, 202-D AMP feature, 153 motions, 44,635 windows). A physical-GPU2 2-env one-update resume completed with termination 0 and PPO update fraction 1.0. Ruff was unavailable in the remote environment. Checkpoint 34600 was load-verified with policy, policy optimizer, discriminator, discriminator optimizer, AMP normalizer, metadata, and iteration states; SHA256 `b61c9eddb5e29e646c28d742869107b86666c1c0da22e3743ba45b99b786e67f`.
- Result: Launched tmux `bfm_amp018_speedtrack_nofilter_gpu2_resume34500_20260909`, PID `1061460`, `CUDA_VISIBLE_DEVICES=2`, work dir `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_amp018_speedtrack_nofilter_resume34500_gpu2_20260909`. An independent JumpServer reconnect proved progress from iteration 34506 through at least 34603. Iterations 34550-34599 had zero termination/crash/fall, PPO update fraction 1.0, value loss `0.214`, planar/vx/backward MAE `0.227/0.147/0.128`, wrong-way `0.055`, AMP reward `0.186`, and discriminator gap `0.802`; this is a healthy early resume, not yet evidence of behavioral improvement.
- Backup/patch: remote `.codex_backups/amp_stage2_rewardtrack_feet_metrics_20260909.before_amp018_speedtrack_nofilter.20260909_182349.py` (SHA256 `b12d5196d2bc24d01891d51ebd18433ac84624bc2a900628b74991735e3257ca`); `/tmp/amp_stage2_amp018_speedtrack_nofilter_20260909.patch` (SHA256 `3d4d29ede337332f76fbd88a8c1d3cab3b0a9bcafc8837910c708b08c9963660`); candidate SHA256 `deb68b06ab88f247957f97351b6bdbc694f55000d8673edbb5343b46b9ca7578`.
- Launch command: `CUDA_VISIBLE_DEVICES=2 python -m humanoidverse.amp_stage2_amp018_speedtrack_nofilter_20260909 --device cuda:0 --gpu-ids single --work-dir logs/amp_stage2_piplus_lse_1gpu_4096env_1m_amp018_speedtrack_nofilter_resume34500_gpu2_20260909 --resume logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_upright_resume32100_20260905/checkpoint_34500.pt --num-envs 4096 --iterations 1000000 --rollout-steps 32 --history-length 8 --command-resample-steps 300 --command-resample-prob 0.75 --command-stand-prob 0.25 --command-turn-prob 0.20 --command-backward-prob 0.40 --command-warmup-steps 20 --command-smoothing 0.10 --latent-stat-max-frames 4096 --latent-reference-size 1024 --latent-prior-weight 0.02 --latent-diagnostic-samples 512 --ppo-epochs 3 --minibatch-size 1024 --learning-rate 5e-6 --target-kl 0.02 --discriminator-learning-rate 5e-6 --amp-weight 0.18 --entropy-coef 0.003 --env-reward-weight 1.0 --locomotion-reward-weight 1.1 --linvel-exp-weight 2.8 --backward-velocity-progress-weight 2.0 --max-episode-length-s 20.0 --seed 1 --save-every 100`.

## 2026-09-09 19:49:12 CST - Switch GPU2 no-filter/no-LoRA Stage2 to five-direction balanced experts

- Context: Physical GPU2 PiPlus LSE Stage2; source `checkpoint_34500.pt` SHA256 `c335ff4152bba8e73d76dfa159116842c07d078cc0fcfd1809ed7a84b3dffc34`; GPU1 baseline unchanged.
- Phenomenon: The prior GPU2 no-filter/no-LoRA run still used the 153-motion `run_with_stand` expert dataset, whose AMP windows were overwhelmingly forward. The user requested the previously generated five-direction equal-window dataset while retaining no filter, no LoRA, and the same 34500 starting point.
- Analysis: Change only the expert distribution so the comparison remains attributable. Keep the no-filter source and the same AMP/velocity/PPO/reward settings; retain the old GPU2 checkpoint 35000 as a rollback lineage.
- Adjustment: Replaced `--expert-dataset` with `dataset/pkl_cmd/piplus_lse_5dir_equal_h8_486w.pkl` (SHA256 `122f76edfd6ae1188973538fe51422235f723e3ddffb2d32060ff062c1e06a66`). It contains 11 motions and exactly 486 valid 8-frame windows each for stand, forward, backward, lateral, and turn (2,430 total). No source, reward, PPO, robot, observation, action, or checkpoint-format change was stacked.
- Rationale: Balanced global AMP sampling gives every motion category 20% expert mass in the current discriminator, directly testing whether the forward-heavy expert prior caused weak backward/side/turn behavior.
- Expected effect: Higher AMP compatibility for backward/lateral/turn commands without sacrificing safety or command tracking. Monitor discriminator adaptation, category behavior, yaw/fast-forward guardrails, foot clearance, smoothness, PPO early-stop, and value loss.
- Rollback: Gracefully stopped tmux `bfm_amp018_speedtrack_nofilter_gpu2_resume34500_20260909`, PID `1061460`; its complete `checkpoint_35000.pt` was load-verified with policy, both optimizers, discriminator, AMP normalizer, metadata, and SHA256 `6fef69ed5dcef315ab9bc553600420eb6404e5c269f60f99541cd174d087b918`. No files were deleted.
- Validation: Candidate CPU dry-run passed with 23 DoF, 202-D AMP features, 11 motions, and 2,430 windows. A physical-GPU0 2-env one-update resume from 34500 completed with termination 0 and PPO update fraction 1.0. The formal checkpoint 34600 was load-verified with all required states and SHA256 `53eb9de3725c72ee6c3df521a6b8c44926dede52f8d0b143f16ff7999115075a`; its metadata records the balanced expert path, `action_filter=none`, and no LoRA fields.
- Result: Launched tmux `bfm_5dir_equal_nofilter_nolora_gpu2_resume34500_20260909`, PID `1068774`, `CUDA_VISIBLE_DEVICES=2`, work dir `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_5dir_equal_nofilter_nolora_resume34500_gpu2_20260909`. Independent JumpServer reconnect proved progress from iteration 34503 through at least 34616. The first 100-window average had zero termination/crash/fall, planar/vx/vy/yaw MAE `0.225/0.145/0.142/0.429`, backward MAE `0.125`, wrong-way `0.053`, AMP reward `0.366`, and discriminator gap `0.560`; resume transients made average PPO update fraction `0.813` and value loss `3.56`, but the latest 34616 row had PPO update fraction `1.0`, value loss `0.120`, planar/backward MAE `0.204/0.097`, wrong-way `0.48%`, and no fatal signature.
- Launch command: `CUDA_VISIBLE_DEVICES=2 python -m humanoidverse.amp_stage2_amp018_speedtrack_nofilter_20260909 --device cuda:0 --gpu-ids single --work-dir logs/amp_stage2_piplus_lse_1gpu_4096env_1m_5dir_equal_nofilter_nolora_resume34500_gpu2_20260909 --resume logs/amp_stage2_piplus_lse_1gpu_4096env_1m_rewardtrack_upright_resume32100_20260905/checkpoint_34500.pt --expert-dataset dataset/pkl_cmd/piplus_lse_5dir_equal_h8_486w.pkl --num-envs 4096 --iterations 1000000 --rollout-steps 32 --history-length 8 --command-resample-steps 300 --command-resample-prob 0.75 --command-stand-prob 0.25 --command-turn-prob 0.20 --command-backward-prob 0.40 --command-warmup-steps 20 --command-smoothing 0.10 --latent-stat-max-frames 2430 --latent-reference-size 1024 --latent-prior-weight 0.02 --latent-diagnostic-samples 512 --ppo-epochs 3 --minibatch-size 1024 --learning-rate 5e-6 --target-kl 0.02 --discriminator-learning-rate 5e-6 --amp-weight 0.18 --entropy-coef 0.003 --env-reward-weight 1.0 --locomotion-reward-weight 1.1 --linvel-exp-weight 2.8 --backward-velocity-progress-weight 2.0 --max-episode-length-s 20.0 --seed 1 --save-every 100`.

## 2026-09-09 23:15:00 CST - Launch frozen-BFM H0W 22-DoF AMP Stage2 on GPU0/1

- Context: User requested the uploaded PiPlus H0W/22-DoF code and `piplus_nowaist_walk_run.pkl` expert dataset, with frozen BFM and no LoRA. The previous GPU1 project run was stopped gracefully; GPU2 remained occupied by an unrelated lineage and was not touched.
- Adjustment: Started tmux `bfm_amp_stage2_piplus_h0w22_gpu01_retry2_20260909` with `CUDA_VISIBLE_DEVICES=0,1`, `HT_BFM_ISOLATE_WORKER_GPU=0`, two torchrun workers, 2048 environments per rank, `--disable-lora`, `--amp-command-matched`, and the H0W BFM checkpoint/robot config. The temporary first launch failed before training because its log directory name did not exist; no process or checkpoint was left by that attempt.
- Validation: Independent SSH reconnect confirmed rank PIDs `1105551`/`1105552` on physical GPU0/GPU1, while GPU2 still only has its pre-existing process. The remote log reached iteration 36 with `lora_grad_norm=0.0`, `bfm_latent_check.projected_norm_mean=16.0`, finite AMP/PPO metrics, and no crash signature. No Stage2 checkpoint exists yet because `--save-every 100`.
- Current metrics (iteration 36): reward `-1.776`, AMP reward `-1.361`, termination rate `0.0823`, planar MAE `0.478`, backward-bin MAE `0.369`; these are expected early-training values and are not yet a performance claim.
- Launch command: `CUDA_VISIBLE_DEVICES=0,1 HT_BFM_ISOLATE_WORKER_GPU=0 /root/gpufree-data/zhuzejian/.conda/envs/env_isaaclab/bin/python -m humanoidverse.amp_stage2 --bfm-checkpoint 'huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)/checkpoint' --expert-dataset dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml --disable-lora --amp-command-matched --device cuda --gpu-ids all --num-envs 2048 --iterations 1000000 --rollout-steps 32 --history-length 8 --command-resample-steps 300 --command-resample-prob 0.75 --command-stand-prob 0.25 --command-turn-prob 0.20 --command-lateral-prob 0.15 --command-warmup-steps 20 --command-smoothing 0.10 --work-dir logs/amp_stage2_piplus_h0w22_gpu01_retry1_20260909`.

- Follow-up 2026-09-09 23:20 CST: independent SSH verification found the same two-worker tmux session alive at iteration 89 and then checkpoint `checkpoint_100.pt` present. Remote `config.json` records `action_dim=22`, `distributed_world_size=2`, `amp_command_matched=true`, `bfm_trainable_parameter_count=0`, and `bfm_lora.enabled=false`; no fatal traceback/NCCL/OOM was found. Iteration 89 metrics: reward `-1.589`, AMP reward `-0.677`, termination `0.081`, planar MAE `0.465`, backward-bin MAE `0.302`, value loss `273.66`. These remain early-training diagnostics rather than a final performance claim.

## 2026-09-10 10:20 CST - Prepare GPU4 mirror and four-card H0W AMP Stage2 launch

- Context: The user expanded the deployment target to GPU4 host `XN-5090-GPU4` and requested the same frozen-BFM/AMP PiPlus 22-DoF Stage2 run with `num_envs=4096` on four cards.
- Adjustment: Mirrored the Stage2 module, H0W robot config, expert dataset/manifest, converter, tests, launch script, and missing Stage1 `config.yaml` into `/data/zhuzejian/HT_BFM`; copied the current simulator IsaacSim implementation so `BFMZERO_ASSET_CACHE_DIR` is honored. Remote hashes for `amp_stage2.py`, robot config, and dataset match GPU11/local (`85fc5f7f...d6f62`, `9c90ab59...eb947`, `0b5edaa2...5a95e`).
- Environment: Installed missing `mujoco`, `loguru`, `easydict`, and `numpy-stl` into `/data/zhuzejian/.conda/envs/env_isaaclab`. The GPU4 runtime requires `PYTHONPATH=/data/zhangrui/ht_urdf`, `HT_URDF_ROOT=/data/zhangrui/ht_urdf/ht_urdf`, writable `TMPDIR=/tmp/zhuzejian_isaaclab`, and `BFMZERO_ASSET_CACHE_DIR=/tmp/zhuzejian_isaaclab/IsaacLab`.
- Validation: GPU4 `tests/test_amp_stage2.py` passed 28/28 with the asset environment. CPU dry-run passed with 22 actions, 194-D AMP features, 15 motions/2,105 frames, projected latent norm 16, and zero BFM trainable parameters. A GPU1 Isaac smoke run completed one PPO update with zero termination/crash/fall; only non-fatal renderer/URDF deprecation warnings remained.
- Resource guardrail: GPU4 physical cards 0,2,3,6,7 are occupied by unrelated `zhangrui` InstinctRL jobs (not this project); only cards 1,4,5 are idle. No unrelated process was stopped or shared. A detached tmux supervisor `bfm_amp_stage2_piplus_h0w22_gpu4_4gpu_wait_20260910` now waits for four cards with <100 MiB usage, then launches the exact `--num-envs 4096 --gpu-ids all` command on those four physical cards. Current GPU11 two-card H0W run remains alive and advanced to iteration 7678/checkpoint 7600, with `lora_grad_norm=0`.

## 2026-09-10 11:45 CST - Compare current GPU11 22DoF frozen-BFM AMP and 23DoF balanced AMP

- GPU11 22DoF lineage: `logs/amp_stage2_piplus_h0w22_gpu01_retry1_20260909`, PIDs `1105551/1105552`, latest parsed iteration `8089`, checkpoint `checkpoint_8000.pt`. Compared iterations `0-99` with `7990-8089`: reward improved `-1.696 -> -0.437`, AMP reward `-0.955 -> -0.045`, planar MAE `0.475 -> 0.447`, and value loss `334.15 -> 2.24`; however termination worsened `0.0820 -> 0.1079`, crash termination `0.0777 -> 0.1077`, discriminator gap grew `6.32 -> 8.66`, and PPO update fraction remained very low (`0.0155 -> 0.0363`). Current backward-bin MAE and nonzero-command metrics are zero, so backward improvement is not demonstrated. Classification: partial reward/tracking improvement but safety and AMP/PPO health are not yet satisfactory.
- GPU11 23DoF lineage: `logs/amp_stage2_piplus_lse_1gpu_4096env_1m_5dir_equal_nofilter_nolora_resume34500_gpu2_20260909`, PID `1068774` on physical GPU2, latest iteration `43433`, checkpoint `checkpoint_43400.pt`. Compared iterations `34500-34599` with `43334-43433`: termination stayed `0`, planar MAE improved `0.224 -> 0.218`, backward MAE `0.125 -> 0.112`, nonzero yaw MAE `0.477 -> 0.273`, value loss `3.996 -> 0.079`, and PPO update fraction `0.804 -> 1.0`; vx MAE slightly regressed `0.145 -> 0.154`, reward rose `0.208 -> 0.220`, and AMP reward remained positive (`0.364 -> 0.338`). Classification: stable with real yaw/backward improvement and a small forward-vx tradeoff.

## 2026-09-10 12:55 CST - Fix 22DoF reset command starvation and resume historical candidate

- Architecture finding: In the original 22DoF loop, a done environment set both `commands[done]` and `command_targets[done]` to zero, then `_mimiclite_resample_mask` waited for `command_warmup_steps=20`. Because crash episodes commonly ended before 20 steps, those environments repeatedly reset into zero-command episodes. This explains the original late-window backward/nonzero command fractions becoming exactly zero and is a data/rollout contract bug, not merely a reward-weight issue.
- Change: Added `_resample_done_command_targets` and `--command-resample-on-reset` (default true) in `humanoidverse/amp_stage2.py`; reset environments now receive a fresh target while their live command remains zero for the reset transition. Added a shape/dtype/non-mutation unit test. Local validation: 29/29 tests, py_compile, and ruff passed. Remote candidate hash: `caaf659d348a7b2ba6917ac3d76d515c4f8c03d14bbdc8f10107297ddf2c3dc2`; remote tests 28/28 and dry-run passed.
- Candidate screening: Fixed headless MuJoCo backward play on original checkpoints 100/2200/4000/8000. Checkpoint 100 was the least damaging screen (no termination in the first 60 steps, though yaw became unstable by 100 steps); checkpoint 2200 had wrong-way vx and QACC warning by 0.46s, checkpoint 4000 terminated/QACC by 0.625s, and checkpoint 8000 had QACC warning by 1.135s. Chose immutable `checkpoint_100.pt` (complete state set, iteration 100, action_dim 22, frozen BFM metadata) as the rollback/resume anchor; this is a conservative historical champion, not a claim of solved behavior.
- Resume: Gracefully stopped the old 22DoF lineage, then launched tmux `bfm_amp_stage2_piplus_h0w22_cmdreset_resume100_20260910` on physical GPU0/1 with `--command-resample-on-reset`, `--command-warmup-steps 0`, policy LR `5e-5 -> 1e-5`, discriminator LR `2e-4 -> 5e-5`, unchanged robot/22DoF/AMP/no-LoRA contract, and resume from `checkpoint_100.pt`. The reset fix restored backward-bin sampling to `~0.167` instead of 0; PPO update fraction initially rose to `0.2-0.66` versus the old `~0.01-0.04`.
- Post-disconnect evidence: Fresh SSH reconnect confirmed both workers on GPU0/1, no NCCL/OOM/traceback, and iteration advanced from `305` to `355`. New `checkpoint_200.pt` and `checkpoint_300.pt` exist; fixed backward play at checkpoint 300 ran 100 steps without QACC warning (one termination), better than all screened original checkpoints. The challenger remains under evaluation; current last-30 metrics at iteration 355 are termination `0.0848`, crash `0.0821`, backward fraction `0.1665`, backward MAE `0.303`, planar MAE `0.482`, and PPO update fraction `0.043`.

## 2026-09-10 16:16:46 CST - Stabilize H0W 22DoF AMP Stage2 from checkpoint 300

- Context: GPU11 tmux bfm_amp_stage2_piplus_h0w22_cmdreset_resume100_20260910; frozen H0W BFM + command encoder + AMP; checkpoint 300
- Phenomenon: At iteration 2201, termination_rate 0.1018, crash_rate 0.0979, planar MAE 0.4996, vx MAE 0.3587, approx_KL 0.0345, clip_fraction 0.4227, value_loss 93.63, discriminator gap 8.75; checkpoint 2200 terminated at fixed backward MuJoCo playback step 100, while checkpoint 300 completed 100 steps without termination.
- Analysis: The reset-command starvation fix restored command coverage, but the resumed encoder still makes overly large PPO/adversarial updates and produces unstable command-conditioned BFM behavior. Checkpoint 300 is the best verified historical point in this lineage; reduce update aggressiveness and command transition speed while retaining frozen BFM, command-matched AMP, and reset resampling.
- Adjustment: policy learning rate: 1e-5 -> 5e-6; discriminator learning rate: 5e-5 -> 2e-5; PPO epochs: 4 -> 2; target KL: 0.04 -> 0.02; AMP weight: 0.06 -> 0.04; command smoothing: 0.10 -> 0.05; resume checkpoint: 2200 -> 300
- Rationale: Lower PPO and discriminator step size addresses high KL/clip/value-loss instability; lower AMP pressure reduces adversarial shock; slower command target transitions reduce abrupt BFM command changes. The checkpoint rollback is supported by fixed playback evidence.
- Expected effect: Termination/crash and KL should decrease, PPO update fraction should stay near 1, and fixed-command velocity overshoot should reduce; watch tracking MAE and AMP score for regression.
- Result: Pending remote resume and fresh-connection verification.
- Files/commands: humanoidverse/amp_stage2.py unchanged; remote launch command and logs/amp_stage2_piplus_h0w22_stable_resume300_20260910

## 2026-09-10 18:58:12 CST - Add minimum foot-separation penalty and resume 23DoF from checkpoint 46900

- Context: GPU11 23DoF lineage amp_stage2_amp018_speedtrack_nofilter_20260909; yaw_stability_resume46200 lineage stopped by user at iteration 46902 after playback review (checkpoint 46600 fixed-command 0.3 test). Pre-stop metrics at iteration 46902: reward 0.259, termination/crash/fall all 0, planar MAE 0.180, vx MAE 0.127, yaw_rate MAE 0.195 (nonzero 0.219), approx_kl 0.0095, clip_fraction 0.118, value_loss 0.048, mimiclite contributions linvel_exp 0.0538 / angvel_z_exp 0.0464 / backward_progress 0.0123.
- Phenomenon: Playback shows the two feet getting too close together (narrow stance), weakening standing/forward-backward stability. No existing reward constrains lateral foot spacing: single_foot_contact gates single support, feet_clearance only constrains swing height.
- Analysis: Nominal PiPlus stance puts the ankle-roll links ~0.163 m apart laterally (hip pitch 0.0475 + hip roll 0.034 per leg, URDF). Add a penalty only when lateral separation falls below a floor of 0.14 m, active only while standing or moving slowly (<0.5 m/s planar, |vy|<0.3, |yaw|<0.5) so lateral stepping and turning gaits are not distorted.
- Adjustment: New reward term foot_separation_penalty = -(min_separation - lateral_distance).clamp_min(0)^2 with weight 20.0 (raw value is squared meters, so the weight is large relative to unit-range trackers: 0.02 m violation ~1.6e-4/step, 0.08 m ~2.6e-3/step at dt 0.02). New CLI: --foot-separation-weight (default None -> 20.0), --foot-min-separation (default None -> 0.14). New diagnostics: lateral_foot_separation_mean, foot_separation_violation_mean, foot_separation_active_fraction. Constructor validates exactly two feet and positive min separation. Unit tests added (5 cases) in tests/test_amp_stage2.py; local suite 34/34, py_compile and ruff pass. All other reward weights unchanged.
- Rationale: The penalty is a gentle floor, not a hard constraint: quadratic in violation so tiny narrowing costs almost nothing while 5-10 cm narrowing becomes meaningful; gating avoids fighting normal turning/lateral gait foot crossing.
- Expected effect: Lateral foot separation during stand/slow phases rises toward >= 0.14 m; standing and forward/backward stability should improve without regressing yaw/planar tracking (watch tracking MAEs, termination rate, and the three new diagnostics).
- Result: Patch applied to remote repo (sha256 432645f9... matches local; v2 fixed a pre-initialized feet_diagnostics_store KeyError caught by the GPU2 smoke run). Smoke passed: iteration 0 metrics show lateral_foot_separation_mean 0.175 at nominal stance, violation 0, penalty contribution 0. New lineage launched on physical GPU2 from checkpoint_46900: tmux bfm_footsep_resume46900_20260910, workdir logs/amp_stage2_piplus_lse_1gpu_4096env_1m_footsep_resume46900_20260910, all other args identical to yaw_stability plus --foot-separation-weight 20 --foot-min-separation 0.14. First check at iteration 46998: termination/crash 0, approx_kl 0.0095, lateral_foot_separation_mean 0.123 (below the 0.14 floor, confirming the observed narrow stance), foot_separation_violation_mean 0.0326, active_fraction 0.61, penalty contribution -3.8e-4 (gentle, as designed). Watch lateral_foot_separation_mean rise toward 0.14+ over the next few hundred iterations while tracking MAEs stay comparable.
- Files/commands: humanoidverse/amp_stage2_amp018_speedtrack_nofilter_20260909.py; tests/test_amp_stage2.py (34/34 pass); remote patches /tmp/amp_stage2_foot_separation_20260910.patch and /tmp/amp_stage2_foot_separation_v2_20260910.patch (git apply on GPU11)

## 2026-09-10 20:11:55 CST - 22DoF H0W optreset_resume300 health check at iteration 2774 + checkpoint 2600 playback screen

- Context: GPU11 tmux bfm_amp_stage2_piplus_h0w22_optreset_resume300_20260910; frozen H0W BFM + command encoder + command-matched AMP, 2048 envs, 2 GPUs, running since 16:24, latest checkpoint 2600 at 19:53.
- Phenomenon: Training-health metrics improved since resume: value_loss 313->74, PPO update fraction 0.02->1.0 (first time stable), yaw MAE 2.46->1.90 (nonzero 2.78->1.74), reward -1.51->-1.19, KL stable ~0.016. But termination rose 6.8%->10.2% (crash 6.7%->9.8%), planar MAE 0.43->0.47, vx MAE 0.28->0.34, clip_fraction 0.25, amp_score -7.7. Fixed-command MuJoCo playback of checkpoint_2600 (200 steps, headless): forward 0.3 -> robot walks BACKWARD (vx -0.4..-1.2), termination=1 by step 50, QACC instability warning at t=1.67s; backward -0.3 -> real fall terminated=True at step 100, yaw MAE 13.5; yaw-left 0.5 -> yaw velocity oscillates +-7..9 rad/s (vs command 0.5), tracking_yaw_rate_mae 10.9. All three directions fail: falls, wrong-way motion, huge joint accelerations.
- Analysis: The lineage is only at 2774/1M iterations (3.5h). PPO update fraction just reached 1.0 for the first time in this lineage's history (previous runs stuck at 0.01-0.13), so the encoder is only now actually learning steadily. amp_score -7.7 means the command-matched discriminator still rejects the policy's motion style strongly, contributing negative AMP reward. clip_fraction 0.25 with KL 0.016 indicates a wide ratio distribution from the PPO updates. Wrong-direction forward response is a long-standing issue in this lineage (also seen at checkpoint 2200 in the 12:55 screen).
- Adjustment: No parameter change yet. Set a review gate at 4000-5000 iterations; candidate optimizations if metrics do not improve: (1) lower PPO clip ratio to 0.15 given clip_fraction 0.25; (2) reduce amp_weight 0.04 -> 0.02 or revisit command-matched AMP given amp_score -7.7; (3) strengthen linvel_exp weight (the H0W amp_stage2.py defaults predate the 23DoF 3.2 fix); (4) add explicit crash/survival shaping if termination keeps rising.
- Rationale: Changing rewards or hyperparameters now would discard the first-ever stable PPO learning phase; the health trajectory (value loss, PPO fraction, yaw) is improving, so give the conservative settings time before intervening.
- Expected effect: By iteration 4000-5000: termination should plateau or fall below 8%, planar MAE should trend below 0.4, and fixed-command playback should show correct-direction forward/backward response without QACC warnings.
- Result: Assessment recorded; no intervention. Next review at iteration 4000-5000 or on user request.
- Files/commands: Remote eval log /tmp/h0w22_eval.log (forward/backward/yaw_left screens of checkpoint_2600)
