I want a bead finding node for a volume stack to find centroids with the following algorithm
1) take multiple z-stacks of various sizes S equally spaced throughout the volume, with some amount of overlap G and do some type of max intensity, min intensity, average intensity etc to flatten these regions. The spacing of such depends on the density of beads where if known G and S can be estimated. 
2) on each flattened image find circles, bright spots, or other feature identifiers for a bead
3) Crop the bead from the raw z stack based on the xy centroid, and find the location of the centroid in z. the location of z should be near the z-stacks region
4) do this for all beads in the volume, to back out the centriod x,y,z which can be sub pixel based on a gaussian of the intensity, in z this gaussian might be skewed due to convolution. 
5) the output is a list of centroids for each possible bead
6) we filter for beads of the correct sizes 