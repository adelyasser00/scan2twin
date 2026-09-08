# CLAUDE.md

## lecturehall dataset (verified 6 Sep 2026)
- 5-col CSV: x, y, z, scan_num (always 0), label (0 static / 1 dynamic)
- COORDINATES ARE CENTIMETRES. Multiply by 0.01.
- right-handed, z up
- pose files: tx ty tz in cm, rx ry rz in RADIANS
- scan 1 pose is effectively identity; scan 1 defines the frame
- scan 2 REQUIRES the full pose2 transform:
    pts_m = (pts_cm * 0.01) @ R.T + (t_cm * 0.01)
    R = Rz @ Ry @ Rx
  Verified: overlap-restricted median NN 0.046 m, 25th pct 0.030 m.
  Identity gives 0.229 m. Rotation without translation gives 0.362 m.
- scanner origins in metres: station 1 ~ (0.009, 0.049, 0.031),
  station 2 = (1.698, -6.458, -0.691). Separation 6.71 m.
- ground truth: scan 1 has 521,923 dynamic points, scan 2 has zero.
- NOTE: labels are TWO-CLASS. No noise class. Surface damage will be
  inflated by however much real sensor noise the carver correctly removes.
  This must be stated in any published number.
