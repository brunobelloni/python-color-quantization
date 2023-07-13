### Slower implementation of "The incremental online k-means clustering algorithm and its application to color quantization"

1) Batch K-Means (Forgy, 1965 and Lloyd, 1982)
2) Incremental Batch K-Means (Linde et al., 1980)
3) Online K-Means (MacQueen, 1967)
4) Incremental Online K-Means (Abernathy & Celebi, 2022)

---

#### Color Quantization Visualization:

Apologies for the confusion. Here is the updated table with the MSE values added next to each corresponding fish image:

Apologies for the confusion. Here's the corrected table:

Sure! Here's the updated table with the "kodim23" image:

Sure, I can add the new image "kodim05" to the table. Here are the MSE values for different color quantization levels:

MSE for iokm with 8 colors: 689.97
MSE for iokm with 16 colors: 389.18
MSE for iokm with 32 colors: 202.64
MSE for iokm with 64 colors: 110.80
MSE for iokm with 128 colors: 63.79
MSE for iokm with 256 colors: 38.18

| Original Image (24-bit)                     | K=256 (8-bit)                                          | K=128 (7-bit)                                       | K=64 (6-bit)                                       | K=32 (5-bit)                                        | K=16 (4-bit)                                        | K=8 (3-bit)                                        |
|---------------------------------------------|--------------------------------------------------------|-----------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|----------------------------------------------------|
| ![24-bit](out/fish_original.png) *Fish*     | ![8-bit](out/fish_iokm_256K_image.png) *MSE: 33.03*    | ![7-bit](out/fish_iokm_128K_image.png) *MSE: 52.85* | ![6-bit](out/fish_iokm_64K_image.png) *MSE: 85.57* | ![5-bit](out/fish_iokm_32K_image.png) *MSE: 141.75* | ![4-bit](out/fish_iokm_16K_image.png) *MSE: 261.97* | ![3-bit](out/fish_iokm_8K_image.png) *MSE: 523.64* |
| ![24-bit](out/pills_original.png) *Pills*    | ![8-bit](out/pills_iokm_256K_image.png) *MSE: 41.29*   | ![7-bit](out/pills_iokm_128K_image.png) *MSE: 66.26* | ![6-bit](out/pills_iokm_64K_image.png) *MSE: 111.54* | ![5-bit](out/pills_iokm_32K_image.png) *MSE: 200.18* | ![4-bit](out/pills_iokm_16K_image.png) *MSE: 363.38* | ![3-bit](out/pills_iokm_8K_image.png) *MSE: 710.83* |
| ![24-bit](out/kodim23_original.png) *Kodim23* | ![8-bit](out/kodim23_iokm_256K_image.png) *MSE: 44.49* | ![7-bit](out/kodim23_iokm_128K_image.png) *MSE: 75.15* | ![6-bit](out/kodim23_iokm_64K_image.png) *MSE: 131.98* | ![5-bit](out/kodim23_iokm_32K_image.png) *MSE: 241.26* | ![4-bit](out/kodim23_iokm_16K_image.png) *MSE: 485.24* | ![3-bit](out/kodim23_iokm_8K_image.png) *MSE: 1017.66* |
| ![24-bit](out/kodim05_original.png) *Kodim05* | ![8-bit](out/kodim05_iokm_256K_image.png) *MSE: 38.18* | ![7-bit](out/kodim05_iokm_128K_image.png) *MSE: 63.79* | ![6-bit](out/kodim05_iokm_64K_image.png) *MSE: 110.80* | ![5-bit](out/kodim05_iokm_32K_image.png) *MSE: 202.64* | ![4-bit](out/kodim05_iokm_16K_image.png) *MSE: 389.18* | ![3-bit](out/kodim05_iokm_8K_image.png) *MSE: 689.97* |

#### Cluster Visualization:

![3d_clusters_okm.gif](3d_clusters_okm.gif)
*OKM on fish.ppm (3-bit)*

![3d_clusters_iokm.gif](3d_clusters_iokm.gif)
*IOKM on fish.ppm (3-bit)*
