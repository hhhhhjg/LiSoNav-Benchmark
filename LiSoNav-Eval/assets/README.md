# LiSoNav-Eval external assets

`lifelong_navigation_sequences/` contains benchmark metadata. The scene and
movable-object assets referenced by that metadata retain their upstream
licenses and are not covered by the benchmark software license.

## HM3D

Obtain HM3D v0.2 through the official Matterport/Habitat access process and
place it at `data/hm3d`. HM3D is not distributed with this repository.

## Curated YCB/HSSD movable objects

Place the YCB/HSSD asset directory at `data/ycb_and_hssd`.

The assets include:

- YCB-derived objects, CC BY 4.0, originally from
  <https://www.ycbbenchmarks.com/>;
- HSSD-derived objects, CC BY-NC 4.0, originally from
  <https://huggingface.co/datasets/hssd/hssd-hab>.

Retain `source_dataset_map.json`, the applicable license notices, source links,
and attribution. HSSD-derived objects are restricted to non-commercial use
unless separate permission is obtained.
