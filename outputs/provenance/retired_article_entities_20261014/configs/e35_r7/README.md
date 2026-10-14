# Fixed-640 e35/r7 training matrix

This directory freezes the first controlled training matrix after the 12,800-
step probes. Every run uses one class, 640x640 input, 640 decoder queries,
`maxDets=300`, seed 0 and global batch 64 on eight GPUs.

The normative SGQS-E implementation and mathematics are commit
`6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a`. `method_provenance.yml`
records canonical-LF hashes for every method-critical file. The launcher
checks those hashes and the fully resolved YAML before it may create a run.

| Run | Initialization | Training interval | Change from fixed-640 VFL |
|---|---|---:|---|
| `deim_core_e35` | common COCO detector tuning checkpoint | epochs 0--34 | MAL + Dense O2O |
| `resume_b0_baseline` | `checkpoint0027.pth` | epochs 28--34 | none |
| `resume_b1_sgqs` | same epoch-27 checkpoint | epochs 28--34 | SGQS only |
| `resume_b2_efgl` | same epoch-27 checkpoint | epochs 28--34 | EFGL only |
| `resume_b3_sgqs_e` | same epoch-27 checkpoint | epochs 28--34 | SGQS + EFGL |
| `resume_c1_small_reg` | same epoch-27 checkpoint | epochs 28--34 | box/local 1.30/1.15 |
| `resume_c2_mal_only` | same epoch-27 checkpoint | epochs 28--34 | MAL only |
| `resume_c3_mosaic_only` | same epoch-27 checkpoint | epochs 28--34 | Mosaic only |
| `resume_c4_deim_core` | same epoch-27 checkpoint | epochs 28--34 | MAL + Dense O2O |

`resume_b0_baseline` is included only to freeze the control identity. The
matching baseline is already running, so the launcher refuses to start it by
default. The three remaining continuation arms must use the exact same source
checkpoint, data, seed, batch and absolute end epoch as B0.

`frozen_inputs.yml` records the SHA-256 identities measured on the shared
server before deployment. The launcher verifies these identities by default;
set `VERIFY_FROZEN_INPUTS=0` only for a deliberately new experiment identity.

The B0--B3 continuation arms are a checkpoint-continuation screen, not a
substitute for the formal from-start B0/B1/B2/B3 experiment. C1--C4 are
diagnostic comparators: they can reject simpler explanations, but they are not
part of the SGQS-E factorial interaction calculation. SGQS survival is explicitly
started at absolute epoch 28. EFGL also starts at epoch 28 and anneals over 2.1
epochs, which is 30% of this seven-epoch continuation; without this explicit
offset, the original from-start EFGL curriculum would already be zero.

This is a translation of the curriculum time origin, not a new loss. All
owners, four SGQS losses and coefficients, non-owner survival threshold,
supervision protection, EFGL edge weights, Matcher, GO/DDF and DN behavior stay
identical to `6fd2c30`. The only allow-listed runtime delta adds an explicit
EFGL start and duration; its defaults reproduce the reference formula exactly.

`deim_core_e35` is a controlled DEIM-core comparison. It preserves the table
model, optimizer, ReLU activation, fixed input and query budget, changing only
to MAL and Dense O2O. It is not the full upstream COCO recipe, which would also
change the optimizer, learning rate and activation.

The resolved DEIM-core `weight_dict` still contains an inherited `loss_vfl`
entry because this repository deep-merges nested YAML dictionaries. It is
inactive: the criterion dispatch list is exactly `['mal', 'boxes', 'local']`,
so no VFL tensor is produced or added to the training loss.

The historical `targeted_e28_r7_training_bundle.tar.gz` is audit input only;
it must not be extracted over the active tree because it contains superseded
criterion, evaluator and launcher files. Its fresh 28-epoch arm is replaced by
`deim_core_e35`; its completed control is replaced by the archived B0 result.
The four still-informative resume controls were re-authored here as C1--C4 on
top of the current, hash-pinned implementation.

All runs use the same transparent launcher, with the config, checkpoint, GPUs,
batch and other runtime parameters visible in the command. See
`SERVER_COMMANDS.md`; the launcher contains no experiment/config switch table.
