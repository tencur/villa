# campaign/all-fixes — every villa defect we fixed, in one branch

Built on upstream `ScrollPrize/villa` main `e0bbb8b` by merging each of our fix branches one at a time (each branch has its own tests and a real-data proof; see the candidate records and https://github.com/tencur/villa-pr-evidence). Branches that conflicted with each other were resolved by hand so both behaviours stay; nothing from upstream was rewritten. Use this branch to train or run inference without the defects below; it is not itself a PR.

| ID | Branch | What it fixes | Upstream PR |
|---|---|---|---|
| C01 | `fix/surface-preflight-public-s3` | surface_preflight reads the public open-data bucket by its s3:// URI without AWS credentials |  |
| C02 | `fix/reduce-stale-partition-cache` | ink inference reduce step reuses a previous run's partition data from /tmp/partition_cache |  |
| C03 | `fix/prepare-16bit-layers` | ink inference prepare step saturates 16-bit layer TIFFs | #1998 |
| C04 | `fix/zarr-cache-per-volume` | ink inference reads another segment's surface volume from a shared S3 disk cache | #1999 |
| C05 | `fix/folder-mode-checkpoint-identity` | tutorial folder-mode inference skips a different model that has the same checkpoint file name |  |
| C06 | `fix/patch-cache-added-segments` | tutorial training silently ignores segments added after the first run (patch cache) |  |
| C07 | `fix/occupancy-index-local-staleness` | vesuvius.predict keeps skipping chunks that arrived after its occupancy index was built |  |
| C08 | `fix/train-patch-cache-label-identity` | vesuvius.find_patches / vesuvius.train keep the old patch list after the labels change |  |
| C09 | `fix/label-zarr-reconvert-edited-image` | the tutorial's label-conversion command skips a label image that was edited |  |
| C10 | `fix/label-zarr-partial-output` | an interrupted or failed label conversion leaves a partial .zarr that the next run skips |  |
| C11 | `fix/reduce-missing-partition` | the reduce step blends and uploads a result while an inference partition is missing |  |
| C12 | `fix/flat-label-depth-window` | labels converted with create_label_zarrs give no supervision when the surface volume is 21 or 28 slices deep |  |
| C13 | `fix/folder-mode-labelled-segment` | the documented folder-mode inference command finds no segment in the documented 9 um layout |  |
| C14 | `fix/train-final-checkpoint` | ink training does not save its final state unless the run ends on the save schedule |  |
| C15 | `fix/predict-overlap-step` | `vesuvius.predict --overlap X` runs with overlap 1 − X |  |
| C16 | `fix/ema-seed-from-loaded-weights` | fine-tuning a loaded checkpoint with EMA on saves a randomly initialised `ema_model`, which inference prefers |  |
| C17 | `fix/resume-keeps-best-checkpoint` | the first validation after a resume replaces the best checkpoint, whatever it scores |  |
| C18 | `fix/blend-incomplete-part-set` | `vesuvius.blend_logits` merges whatever parts it finds, so a missing part leaves an empty slab |  |
| C19 | `fix/listed-segment-missing` | a segment named in a training config but absent from `segments_path` is dropped without a message |  |
| C20 | `fix/blend-cache-incomplete-part` | a blend run while inference is still writing freezes an incomplete patch list into the logits store |  |
| C21 | `fix/train-previews-without-validation` | `train_previews/` stays empty when the run has no validation set (the tutorial's case) |  |
| C22 | `fix/predict-hf-temp-model-cleanup` | every `vesuvius.predict --model_path hf://...` run leaves a full model copy in /tmp |  |
| C23 | `fix/finalize-uncovered-voxels` | finalize turns voxels that no patch covered into probability 0.5, foreground below threshold 0.5 |  |
| C24 | `fix/patch-writer-late-error` | `vesuvius.predict` exits 0 when the last patch writes fail |  |
| C25 | `fix/blend-bbox-cache-stale` | part boxes cached by one blend decide the parts of a later inference run in the same folder |  |
| C26 | `fix/train-spatial-switches` | `vesuvius.train --no-spatial` and `--rotation-axes` are accepted, logged and ignored |  |
| C27 | `fix/compute-st-delete-intermediate` | `vesuvius.compute_st --delete-intermediate` deletes the final results and reports success |  |
| C28 | `fix/zarr-tasks-zarr3-remaining` | four `vesuvius.zarr_tasks` tasks crash under the zarr 3.2.1 the package locks |  |
| C29 | `fix/edt-dilate-seams` | `vesuvius.zarr_tasks --task edt-dilate` paints shells on every chunk face and stops at seams |  |
| C30 | `fix/train-metrics-num-classes` | `vesuvius.train` validation scores 3-class targets as 2 classes |  |
| C31 | `fix/bg-sampling-ignore-label` | find_patches/train ignore `ignore_label` and `bg_sampling_enabled` |  |
| C37 | `fix/ome-zarr-resolution` | dataset_config.ome_zarr_resolution never reaches the training dataset |  |
| C38 | `fix/binary-label-grayscale` | 0/255 binary labels train BCE against 255 (docs say grayscale is fine) |  |
| C39 | `fix/unlabeled-target-ignore` | missing per-task label trains that task as background |  |
| C40 | `fix/trace-final-save-sentinel` | single-point tracer final save writes empty vertices as -2/-4 |  |
| C43 | `fix/fiber-inference-precision` | fiber_trace_3d.infer ignores --inference-precision (checkpoint's precision wins) |  |
| C44 | `fix/compute-st-integration-smoothing` | vesuvius.compute_st default writes rank-1 tensors (no integration smoothing) |  |
| C45 | `fix/full3d-labels-within-supervision` | ink full_3d projects unsupervised ink (incl. held-out validation ink) as training targets |  |
| C46 | `fix/semi-supervised-held-out-validation` | mean-teacher trainers validate on training patches (and unlabeled patches as background) |  |
| C47 | `fix/prepare-inference-layer-window` | optimized_inference prepare -> inference re-applies the layer window to the cropped zarr | #1997 |
| C48 | `fix/merge-winding-field-chunk-overwrite` | merge_winding_field level-0/1 tiles overwrite shared chunks with zero padding |  |
| C49 | `fix/predict-logits-not-activations` | vesuvius.predict stores eval-time activated outputs as logits |  |
| C50 | `fix/cross-frame-full-scan-snap` | cross_frame full-resolution scan emits patches that miss their foreground |  |
| C51 | `fix/predict3d-resume-checkpoint-mismatch` | lasagna predict3d resumes an existing output with a different checkpoint |  |
| C52 | `fix/train-validation-on-logits` | vesuvius.train validation runs on activated outputs (training side of C49's root cause) |  |
| C53 | `fix/find-patches-plain-array-level` | find_valid_patches treats a bare zarr array as the requested pyramid level |  |
| C54 | `fix/final-config-patch-size` | pretrained-backbone / Primus checkpoints carry no patch_size; predict infers at 128^3 |  |
| C55 | `fix/train-yaml-keys-scheduler-clip-amp-seed` | YAML scheduler / gradient_clip / amp_dtype / no_amp / seed never reach the trainer |  |
| C56 | `fix/displacement-tta-prior-vectors` | copy-model TTA flips the volume but not the direction-prior vectors |  |
| C57 | `fix/ddp-checkpoint-model-config` | DDP training saves checkpoints without model_config while claiming it is embedded |  |
| C59 | `fix/label-version-keeps-validation-mask` | label_version pinning drops an unversioned validation mask |  |
| C60 | `fix/train-loss-override` | vesuvius.train --loss never reaches the trainer |  |
| C61 | `fix/finalize-multiclass-encoding` | multiclass finalize rescales each chunk by its own min/max |  |
| C62 | `fix/finetune-mae-checkpoint-load` | finetune_mae_unet silently trains from scratch (torch weights_only default) |  |
| C63 | `fix/train-seed-reproducible` | vesuvius.train --seed does not make runs reproducible |  |
| C64 | `fix/patch-grid-tail` | training patch grids never cover the remainder strip of each axis |  |
| C65 | `fix/find-patches-any-target-label` | find_patches drops volumes labelled only for a non-first target |  |
| C66 | `fix/semi-supervised-ddp-shards` | mean-teacher trainers give every DDP rank identical batches |  |
| C68 | `fix/checkpoint-history-double-append` | checkpoint rotation keeps two recent epochs instead of three |  |
| C96 | `fix/opencv-pixel-limit` | optimized_inference sets `OPENCV_IO_MAX_IMAGE_PIXELS=0`, which OpenCV treats as a zero-pixel limit: every `cv2.imread` in the package fails |  |
| Cxx | `fix/copy-grow-dense-conditioning` | (see branch; C32-C36 family) |  |
| Cxx | `fix/mae-only-spatial-and-intensity` | (see branch; C32-C36 family) |  |
| Cxx | `fix/mean-teacher-ema-decay` | (see branch; C32-C36 family) |  |
| Cxx | `fix/predict-zscore-per-patch` | (see branch; C32-C36 family) |  |
| Cxx | `fix/train-config-trainer` | (see branch; C32-C36 family) |  |

Superseded or duplicated upstream (kept for completeness): C15 (#1989 draft), C53 (#1980), C56 (#1796), C61 (#1825).

Not included (verified leads without code yet): C69–C95; see candidates/.

Test status at this commit (rackpi6, Python 3.14, torch CPU): `vesuvius/tests` 1113 passed, 10 skipped; 3 failures and 4 collection errors are pre-existing on upstream main in this environment (test_cross_frame_realdata real-data scan, test_zarr3_compat voxelize needs trimesh, tifxyz_label_transfer test_planar default radius, neural_tracing tests import `lasagna`). `ink-detection/optimized_inference/tests` 33 passed. Per-merge logs: see the campaign's gpu-campaign/evidence/allfix_merge_log*.txt.