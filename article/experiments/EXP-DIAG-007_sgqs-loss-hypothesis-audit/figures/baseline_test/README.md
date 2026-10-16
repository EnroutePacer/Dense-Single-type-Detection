# SGQS e28--e34 Baseline_test Top-K / postprocess visualization

- Checkpoints: EMA states from checkpoint0028.pth through checkpoint0034.pth.
- Images: all 13 images under `D:/Baseline_test/DATA`.
- Encoder Top-300 is a diagnostic prefix only; the architecture actually selects Top-640.
- Decoder query count is exactly 640 for every image and checkpoint.
- This is a one-class model. Repository PostProcessor sorts sigmoid final scores, applies Top-K,
  rescales boxes, and applies no NMS or threshold. Therefore `PostProcessor all 640` is the
  score-sorted 640-query final output; `Final Top-300` is the formal maxDets=300 prefix.
- All 91 image/checkpoint states were checked against the repository PostProcessor: scores and
  boxes agree exactly (maximum absolute error 0.0, including tied-score Top-K ordering).
- `score>0.5` is shown only to explain visual density and deployment-style displays; it is not
  the COCO AP-small rule.
- Hybrid-small is a predicted-box descriptor at the 640 input scale:
  `area < 32^2 OR short side < 16`. It is not COCO AP-small ground truth.

## Outputs

- `aggregate_trajectory.png`: per-image trajectories and Encoder source-level shares.
- `baseline_topk_summary.csv`: exact counts and score cutoffs for 13 x 7 states.
- Contact sheets:

  - `wired_image_10.png_e28_e34.png` (wired/image (10).png)
  - `wired_image_11.png_e28_e34.png` (wired/image (11).png)
  - `wired_image_12.png_e28_e34.png` (wired/image (12).png)
  - `wired_image_9.png_e28_e34.png` (wired/image (9).png)
  - `wireless_image_1.png_e28_e34.png` (wireless/image (1).png)
  - `wireless_image_2.png_e28_e34.png` (wireless/image (2).png)
  - `wireless_image_3.png_e28_e34.png` (wireless/image (3).png)
  - `wireless_image_4.png_e28_e34.png` (wireless/image (4).png)
  - `wireless_image_5.png_e28_e34.png` (wireless/image (5).png)
  - `wireless_image_6.png_e28_e34.png` (wireless/image (6).png)
  - `wireless_image_7.png_e28_e34.png` (wireless/image (7).png)
  - `wireless_image_8.png_e28_e34.png` (wireless/image (8).png)
  - `wireless_image.png_e28_e34.png` (wireless/image.png)
