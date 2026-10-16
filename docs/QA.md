# Verification Record

## Initial teaching pipeline

- 29 automated tests passed; after adding deployment-shift tests, 37 passed.
- Two expected Gymnasium warnings: observation Box uses infinite bounds for unconstrained velocities.
- Ruff lint/format and JavaScript syntax checks passed.
- Browser tested at desktop 1440 × 1000 and mobile 390 × 844 viewport settings; no document-level horizontal overflow on mobile.
- Fresh inference through the UI: BC256, OOD, seed 42001 completed in 53 control steps (2.65 s simulated time).
- Failed DAgger replay 20032 selectable; timeline scrub to step 50 displays 2.50 s.
- Playback-speed switching, loss-curve switching, actual MuJoCo GIF dialog open/close verified.
- Browser page-error collection during interaction reported no page errors.
- Screenshots: `artifacts/media/dashboard-desktop.png`, `artifacts/media/dashboard-mobile.png`.

Screenshots are evidence of that tested UI state, not evidence that every future modification remains correct. Repeat relevant checks after changes.

## Frozen study v1

- Completed all 15,000 closed-loop trials and all 25 training runs.
- Independently recomputed summary means, sample SDs and the primary contrast from complete trial records.
- Re-executed the first success/failure in every model/seed/condition group: 271 replays matched all stored scalar outcomes exactly.
- Verified all 17 frozen source/configuration files against their recorded SHA-256 values.
- Research page: full 30-row seed table expands/collapses, paired SVG chart loads, desktop and 390 × 844 mobile layout checked. Wide tables scroll inside their container; document does not overflow horizontally.
- Screenshots in `artifacts/media/research-desktop.png` and `artifacts/media/research-mobile.png` capture tested states. The desktop image predates the later paired-trace section; mobile includes it.
- External robosuite/Panda integration remains unverified while dependency installation is in progress. Its smoke script is not counted among passed integration tests.

## Prospective execution-filter follow-up

- Five added unit/identity-equivalence tests passed.
- Completed 4,500 newly reserved trials with frozen model weights.
- Recomputed complete-case summaries and verified checkpoint/frozen-source hashes.
- Exactly re-executed three paired cases (six episodes): 50ms repair, clean regression, 100ms regression.
- Actual MuJoCo paired rendering was visually inspected; shows final state held after success, with explicit time and status labels.
- Combined automated run: 42 passed, with the same two documented Gymnasium bound warnings.
- Updated research page verifies the +30.6 pp follow-up value, all nine execution/condition table rows, and the actual paired GIF loading at 1280 × 554.

## Panda standard-task replication (completed)

- Resolved the original enum membership failure by pinning only the isolated Panda environment to MuJoCo 3.3.7; minimal before/after reproduction retained.
- Five isolated integration tests passed. Original environment and its 42 tests remain separate.
- Completed all 144 expert trajectories (7,963 transitions), five training runs, 1,500 learned-policy trials and 100 expert/random baseline trials.
- Recomputed complete summaries, verified source/dataset/model hashes and split disjointness, then exactly reproduced 55 automatically selected held-success/failure traces.
- Actual MuJoCo Panda rendering visually inspected; visual meshes enabled, collision display hidden, frontview camera used. These are display choices only, not physics or policy changes.
- Earlier notes saying the external integration was unverified describe earlier work states; they are superseded by this completed record. This is local verification, not independent third-party adoption.
- Final Panda report UI: six metric rows and -6.4 pp primary contrast verified in the browser; actual GIF loaded; desktop and 390-pixel mobile views inspected with no document horizontal overflow.
- Final live API check after restarting the local service: fresh BC256 OOD scene 42002 succeeded in 76 steps. It is not included in the frozen evaluation.
- Panda display GIF samples every other state at 10 fps from the 20 Hz control trajectory; the final state is included. The unchanged raw replay contains every control step.
- Final checks: original/follow-up suite 42 passed; Panda suite 5 passed; dependency check, Ruff lint/format, JavaScript syntax, frozen-source checks, full summary audits and selected exact replays passed. Two original Gymnasium infinite-observation-bound warnings remain documented.
