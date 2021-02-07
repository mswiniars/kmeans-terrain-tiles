# Terrain-tiles data analysis project
The main purpose of this project is to design and implement a measurement system that analyzes the data on terrain height differentiation by selecting groups of areas with the highest growth in Europe. The height increase in a given location is being measured with 10 measurement points. After decoding the data we are clustering it to 5 groups of areas with respect to the average value of the height increase.

## Dataset
In this project we are using terrarium format of the dataset terrain-tiles: https://registry.opendata.aws/terrain-tiles/, which is a global dataset providing bare-earth terrain heights, tiled for easy usage and provided on S3.

## Measurement system
The measurement system is built as follows: 
 - indices of tiles for Europe region are being generated with respect to zoom and Europe boundary geographical coordinates;
 - using fixed indices of tiles the data is being downloaded from s3 bucket and saved on local storage;
 - downloaded data is being decoded to height profile and fed to KMeans algorithm with 5 clusters;
 - after the data is clustered we visualize it as a map.

 ## Results
 We compare both execution time on laptop Lenovo Pavilion 14 with Intel Core i5-8250U 1.6GHz CPU and AWS EC2 t2.2xlarge instance with 8 CPUs of Intel Xeon E5-2686 2.30GHz and 8 Cores per socket. We measure execution time for different zoom ratios. While the zoom ratio increases, the number of images and data being processed by KMeans algorithm grows rapidly. For example for zoom 3 we get 589824 data points being consumed by KMeans algorithm. On the contrary for zoom 7 we get 37748736 number of points.

 | Zoom ratio | Intel Core i5-8250U | AWS EC2 t2.2xlarge |
 | :---: | :---: | :---: |
 | 3 | 15.5s | 6.5s | 
 | 5 | 82.6s | 34.4s |
 | 7 | N/A | 443.20s |
 | 10 | N/A | TBD |

 Example result map for zoom 5:
 ![alt text](https://github.com/mswiniars/results/blob/master/result_5.png?raw=true)