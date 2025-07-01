package core

import "math"

// DistanceFunc computes the distance between two vectors.
// a: the first vector.
// b: the second vector.
// Returns the computed distance as a float64.
type DistanceFunc func(a, b []float32) float64

// EuclideanDistance computes the Euclidean distance between two vectors
// The library is rebuilt to "bring your own" distance functions, however,
// This naive implementation of Euclidean distance is the default and is used if no distance function is provided.
// Mainly for testing purposes.
func Euclidean(a, b []float32) float64 {
	sum := 0.0
	for i := range a {
		sum += float64(a[i]-b[i]) * float64(a[i]-b[i])
	}
	return math.Sqrt(sum)
}
