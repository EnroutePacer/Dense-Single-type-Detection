# Fixed-128 SGQS replay audit

> Per-GT independent best-IoU coverage only; it is not one-to-one COCO TP/FN matching.

## EMA COCO subset metrics

| epoch | APs@300 | ARs@300 | APs@640 | ARs@640 |
|---:|---:|---:|---:|---:|
| 28 | 0.609321 | 0.614884 | 0.970077 | 0.976047 |
| 29 | 0.687327 | 0.688605 | 0.881637 | 0.890698 |
| 30 | 0.656896 | 0.663488 | 0.938852 | 0.948837 |
| 31 | 0.607165 | 0.609767 | 0.958388 | 0.967209 |
| 32 | 0.605198 | 0.607907 | 0.962618 | 0.967674 |
| 33 | 0.602446 | 0.602326 | 0.963954 | 0.968140 |
| 34 | 0.602340 | 0.602093 | 0.964059 | 0.967907 |

## Cohort sizes

- coco_small: 430
- hybrid_small: 7,314
- hybrid_area_only: 2,985
- hybrid_short_only: 261
- hybrid_both: 4,068

## coco_small coverage

### IoU >= 0.5

| epoch | dense all | dense Top-640 | decoder all | decoder Top-300 |
|---:|---:|---:|---:|---:|
| 28 | 426 (99.07%) | 426 (99.07%) | 426 (99.07%) | 266 (61.86%) |
| 29 | 426 (99.07%) | 389 (90.47%) | 390 (90.70%) | 298 (69.30%) |
| 30 | 426 (99.07%) | 417 (96.98%) | 417 (96.98%) | 288 (66.98%) |
| 31 | 426 (99.07%) | 425 (98.84%) | 425 (98.84%) | 265 (61.63%) |
| 32 | 426 (99.07%) | 426 (99.07%) | 426 (99.07%) | 265 (61.63%) |
| 33 | 426 (99.07%) | 426 (99.07%) | 426 (99.07%) | 263 (61.16%) |
| 34 | 426 (99.07%) | 426 (99.07%) | 426 (99.07%) | 263 (61.16%) |

Transitions at decoder Top-300:

- 28_to_29: gained=49, lost=17, net=32; gain source={'final_rank': 49}; loss destination={'encoder_top640': 15, 'final_rank': 2}; gain rank median 359.0->259.0; loss rank median 250.0->641.0; gain density={'gt300': 49}; loss density={'gt300': 17}
- 28_to_30: gained=35, lost=13, net=22; gain source={'final_rank': 35}; loss destination={'final_rank': 7, 'encoder_top640': 6}; gain rank median 342.0->275.0; loss rank median 270.0->345.0; gain density={'gt300': 35}; loss density={'gt300': 13}
- 30_to_34: gained=11, lost=36, net=-25; gain source={'encoder_top640': 6, 'final_rank': 5}; loss destination={'final_rank': 36}; gain rank median 641.0->231.0; loss rank median 271.5->338.5; gain density={'gt300': 11}; loss density={'gt300': 36}
### IoU >= 0.75

| epoch | dense all | dense Top-640 | decoder all | decoder Top-300 |
|---:|---:|---:|---:|---:|
| 28 | 417 (96.98%) | 417 (96.98%) | 426 (99.07%) | 266 (61.86%) |
| 29 | 417 (96.98%) | 368 (85.58%) | 390 (90.70%) | 298 (69.30%) |
| 30 | 419 (97.44%) | 404 (93.95%) | 417 (96.98%) | 288 (66.98%) |
| 31 | 420 (97.67%) | 411 (95.58%) | 425 (98.84%) | 265 (61.63%) |
| 32 | 416 (96.74%) | 414 (96.28%) | 426 (99.07%) | 265 (61.63%) |
| 33 | 416 (96.74%) | 416 (96.74%) | 426 (99.07%) | 263 (61.16%) |
| 34 | 417 (96.98%) | 417 (96.98%) | 426 (99.07%) | 263 (61.16%) |

Transitions at decoder Top-300:

- 28_to_29: gained=49, lost=17, net=32; gain source={'final_rank': 49}; loss destination={'encoder_top640': 15, 'final_rank': 2}; gain rank median 359.0->259.0; loss rank median 250.0->641.0; gain density={'gt300': 49}; loss density={'gt300': 17}
- 28_to_30: gained=35, lost=13, net=22; gain source={'final_rank': 35}; loss destination={'final_rank': 7, 'encoder_top640': 6}; gain rank median 342.0->275.0; loss rank median 270.0->345.0; gain density={'gt300': 35}; loss density={'gt300': 13}
- 30_to_34: gained=11, lost=36, net=-25; gain source={'encoder_top640': 6, 'final_rank': 5}; loss destination={'final_rank': 36}; gain rank median 641.0->231.0; loss rank median 271.5->338.5; gain density={'gt300': 11}; loss density={'gt300': 36}

## hybrid_small coverage

### IoU >= 0.5

| epoch | dense all | dense Top-640 | decoder all | decoder Top-300 |
|---:|---:|---:|---:|---:|
| 28 | 7164 (97.95%) | 7041 (96.27%) | 7075 (96.73%) | 5501 (75.21%) |
| 29 | 7157 (97.85%) | 6344 (86.74%) | 6394 (87.42%) | 5487 (75.02%) |
| 30 | 7158 (97.87%) | 6630 (90.65%) | 6660 (91.06%) | 5492 (75.09%) |
| 31 | 7158 (97.87%) | 6928 (94.72%) | 6966 (95.24%) | 5512 (75.36%) |
| 32 | 7170 (98.03%) | 7055 (96.46%) | 7085 (96.87%) | 5519 (75.46%) |
| 33 | 7173 (98.07%) | 7086 (96.88%) | 7115 (97.28%) | 5524 (75.53%) |
| 34 | 7178 (98.14%) | 7101 (97.09%) | 7128 (97.46%) | 5528 (75.58%) |

Transitions at decoder Top-300:

- 28_to_29: gained=510, lost=524, net=-14; gain source={'final_rank': 475, 'encoder_top640': 30, 'dense_geometry': 3, 'decoder_geometry': 2}; loss destination={'encoder_top640': 457, 'decoder_geometry': 3, 'final_rank': 63, 'dense_geometry': 1}; gain rank median 367.0->250.0; loss rank median 181.5->641.0; gain density={'gt300': 506, '101_300': 4}; loss density={'gt300': 521, '101_300': 3}
- 28_to_30: gained=444, lost=453, net=-9; gain source={'final_rank': 404, 'encoder_top640': 37, 'dense_geometry': 2, 'decoder_geometry': 1}; loss destination={'encoder_top640': 346, 'final_rank': 101, 'decoder_geometry': 2, 'dense_geometry': 4}; gain rank median 350.0->259.5; loss rank median 180.0->641.0; gain density={'gt300': 439, '101_300': 5}; loss density={'gt300': 446, '101_300': 7}
- 30_to_34: gained=429, lost=393, net=36; gain source={'encoder_top640': 298, 'decoder_geometry': 3, 'final_rank': 120, 'dense_geometry': 8}; loss destination={'final_rank': 370, 'encoder_top640': 18, 'decoder_geometry': 4, 'dense_geometry': 1}; gain rank median 641.0->188.0; loss rank median 256.0->344.0; gain density={'gt300': 419, '101_300': 10}; loss density={'gt300': 384, '101_300': 9}
### IoU >= 0.75

| epoch | dense all | dense Top-640 | decoder all | decoder Top-300 |
|---:|---:|---:|---:|---:|
| 28 | 6929 (94.74%) | 6820 (93.25%) | 6925 (94.68%) | 5408 (73.94%) |
| 29 | 6934 (94.80%) | 6058 (82.83%) | 6246 (85.40%) | 5386 (73.64%) |
| 30 | 6947 (94.98%) | 6381 (87.24%) | 6525 (89.21%) | 5403 (73.87%) |
| 31 | 6948 (95.00%) | 6704 (91.66%) | 6823 (93.29%) | 5422 (74.13%) |
| 32 | 6946 (94.97%) | 6840 (93.52%) | 6940 (94.89%) | 5428 (74.21%) |
| 33 | 6947 (94.98%) | 6873 (93.97%) | 6970 (95.30%) | 5436 (74.32%) |
| 34 | 6949 (95.01%) | 6888 (94.18%) | 6980 (95.43%) | 5439 (74.36%) |

Transitions at decoder Top-300:

- 28_to_29: gained=500, lost=522, net=-22; gain source={'final_rank': 467, 'encoder_top640': 27, 'decoder_geometry': 1, 'dense_geometry': 5}; loss destination={'encoder_top640': 449, 'final_rank': 65, 'dense_geometry': 7, 'decoder_geometry': 1}; gain rank median 367.5->250.0; loss rank median 181.5->641.0; gain density={'gt300': 498, '101_300': 2}; loss density={'gt300': 515, '101_300': 6, 'le100': 1}
- 28_to_30: gained=444, lost=449, net=-5; gain source={'final_rank': 395, 'encoder_top640': 34, 'dense_geometry': 12, 'decoder_geometry': 3}; loss destination={'encoder_top640': 341, 'final_rank': 100, 'dense_geometry': 6, 'decoder_geometry': 2}; gain rank median 352.5->256.0; loss rank median 175.0->641.0; gain density={'gt300': 439, '101_300': 5}; loss density={'gt300': 441, '101_300': 8}
- 30_to_34: gained=416, lost=380, net=36; gain source={'encoder_top640': 290, 'final_rank': 115, 'dense_geometry': 9, 'decoder_geometry': 2}; loss destination={'final_rank': 355, 'dense_geometry': 7, 'encoder_top640': 15, 'decoder_geometry': 3}; gain rank median 641.0->184.0; loss rank median 255.5->344.0; gain density={'gt300': 410, 'le100': 1, '101_300': 5}; loss density={'gt300': 373, '101_300': 7}
