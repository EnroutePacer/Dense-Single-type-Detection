# Bundled bbox COCO reference

`coco.py` and `cocoeval.py` are from the **pycocotools 2.0.11** distribution,
whose upstream is https://github.com/ppwwyyxx/cocoapi (the maintained PyPI fork
of https://github.com/cocodataset/cocoapi). Copyright and redistribution terms
are preserved in [LICENSE.txt](LICENSE.txt). Only LF/trailing-whitespace
normalization is applied to those two upstream Python files.

Original source SHA256 before whitespace normalization:

- coco.py: `4a234fc502c6e622f044747312d6a696652545f527e766c98bedf5aa4ca77f41`
- cocoeval.py: `988485f79624cd42ceed9335d284df4cdcdf7b3d583e052fcbed2dd6bccf0267`

The repository's `mask.py` implements **bbox-only** NumPy IoU, including COCO's
crowd denominator (detection area). It does not approximate segmentation/RLE.
The original score sorting, one-to-one matching, crowd/area ignore semantics,
101-point interpolation and accumulation stay in upstream `cocoeval.py`.
The adapter in `../coco_backend.py` reports the backend and fixes the summary
budget to the configured maxDets instead of upstream's hardcoded 100.

This is used only when neither existing faster-coco-eval nor pycocotools can
be imported. It performs no pip installation, compilation, network access or
global module replacement. It is slower than the compiled backend; the normal
single-type bbox task (`return_masks=False`, `iou_types=['bbox']`) is supported.
