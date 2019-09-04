# DigiPath_MLTK : Mayo script



## 1. Patch extraction from svs or tiff files to tfrecords
```
	
python3 Create_ImagePatches.py -i <input file (path to svs or tiff file)> -p <path to patch directory> -o <path to tfrecords> -s <image patch size> -l <input level> -t <threshold value to exclude background from tissue on grey scale> -a < percent of tissue area using threshold> -m < Mean pixel cutoff(grey scale)> -d < standard deviation pixel cutoff(grey scale)>

Example:
python3 Create_ImagePatches.py -i $samp -p $PATCH_DIR -o $TF_DIR -s $Patch_size -l 4 -t 240 -a 0.4 -m 220 -d 5




```
