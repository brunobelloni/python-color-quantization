### Slower implementation of "The incremental online k-means clustering algorithm and its application to color quantization"

1) Batch K-Means (Forgy, 1965 and Lloyd, 1982)
2) Incremental Batch K-Means (Linde et al., 1980)
3) Online K-Means (MacQueen, 1967)
4) Incremental Online K-Means (Abernathy & Celebi, 2022)

---

#### Color Quantization Visualization:

| Original Image (24-bit)           | K=8 (3-bit)                     | K=16 (4-bit)                     | K=32 (5-bit)                     |
|-----------------------------------|---------------------------------|----------------------------------|----------------------------------|
| ![24-bit](out/original_image.png) | ![3-bit](out/iokm_8K_image.png) | ![4-bit](out/iokm_16K_image.png) | ![5-bit](out/iokm_32K_image.png) |


#### Cluster Visualization:

![3d_clusters_okm.gif](3d_clusters_okm.gif)
*OKM on fish.ppm (3-bit)*

![3d_clusters_iokm.gif](3d_clusters_iokm.gif)
*IOKM on fish.ppm (3-bit)*
