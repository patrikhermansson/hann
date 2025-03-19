package core

/*
#cgo CFLAGS: -O2 -mavx
#cgo LDFLAGS: -lm
#include "simd_distance.h"
*/
import "C"
import (
	"unsafe"
)

// Euclidean computes the Euclidean (L2) distance between two input vectors.
func Euclidean(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		panic("vectors must not be empty")
	}
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}
	n := C.size_t(len(a))
	return float64(C.simd_euclidean(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
}

// SquaredEuclidean computes the squared Euclidean distance between two input vectors.
func SquaredEuclidean(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		panic("vectors must not be empty")
	}
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}
	n := C.size_t(len(a))
	return float64(C.simd_squared_euclidean(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
}

// Manhattan computes the Manhattan (L1) distance between two input vectors.
func Manhattan(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		panic("vectors must not be empty")
	}
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}
	n := C.size_t(len(a))
	return float64(C.simd_manhattan(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
}

// CosineDistance computes the cosine distance between two input vectors.
func CosineDistance(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		panic("vectors must not be empty")
	}
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}
	n := C.size_t(len(a))
	return float64(C.simd_cosine_distance(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
}

// AngularDistance computes the angular distance (in radians) between two input vectors using SIMD.
// It calls the AVX-accelerated C implementation.
func AngularDistance(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		panic("vectors must not be empty")
	}
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}
	n := C.size_t(len(a))
	return float64(C.simd_angular_distance(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		n,
	))
}
