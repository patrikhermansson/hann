//go:build ignore
// +build ignore

package main

import (
	"os"

	"github.com/patrikhermansson/hann/core"
	"github.com/patrikhermansson/hann/example"
	"github.com/patrikhermansson/hann/rpt"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	// Set the logger to output to the console.
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Using RPT index with GIST dataset
	RPTIndexGIST()
}

func RPTIndexGIST() {
	factory := func() core.Index {
		dimension := 960
		leafCapacity := 10
		candidateProjections := 3
		parallelThreshold := 100
		probeMargin := 0.15
		return rpt.NewRPTIndex(dimension, leafCapacity, candidateProjections, parallelThreshold,
			probeMargin)
	}

	example.RunDataset(factory, "gist-960-euclidean",
		"example/data/nearest-neighbors-datasets-large", 100, 5, 5)
}
