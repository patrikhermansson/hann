//go:build ignore
// +build ignore

package main

import (
	"os"

	"github.com/patrikhermansson/hann/core"
	"github.com/patrikhermansson/hann/example"
	"github.com/patrikhermansson/hann/pqivf"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Set the logger to output to the console.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Using PQIVF index with GIST dataset
	PQIVFIndexGIST()
}

func PQIVFIndexGIST() {
	factory := func() core.Index {
		dimension := 960
		coarseK := 16
		numSubquantizers := 8
		pqK := 256
		kMeansIters := 10
		return pqivf.NewPQIVFIndex(dimension, coarseK, numSubquantizers, pqK, kMeansIters)
	}

	example.RunDataset(factory, "gist-960-euclidean",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}
