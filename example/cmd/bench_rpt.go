//go:build ignore
// +build ignore

package main

import (
	"net/http"
	_ "net/http/pprof"
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

	// Start the pprof HTTP server on port 6060.
	// This will expose profiling endpoints at /debug/pprof/
	go func() {
		log.Info().Msg("Starting pprof server on :6060")
		if err := http.ListenAndServe("localhost:6060", nil); err != nil {
			log.Error().Err(err).Msg("pprof server failed")
		}
	}()

	// Benchmarking RPT index with FashionMNIST and SIFT datasets
	BenchRPTIndexFashionMNIST()
	BenchRPTIndexSIFT()
}

func BenchRPTIndexFashionMNIST() {
	factory := func() core.Index {
		dimension := 784
		leafCapacity := 10
		candidateProjections := 3
		parallelThreshold := 100
		probeMargin := 0.15
		return rpt.NewRPTIndex(dimension, leafCapacity, candidateProjections, parallelThreshold,
			probeMargin)
	}

	example.RunDataset(factory, "fashion-mnist-784-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}

func BenchRPTIndexSIFT() {
	factory := func() core.Index {
		dimension := 128
		leafCapacity := 10
		candidateProjections := 3
		parallelThreshold := 100
		probeMargin := 0.15
		return rpt.NewRPTIndex(dimension, leafCapacity, candidateProjections, parallelThreshold,
			probeMargin)
	}

	example.RunDataset(factory, "sift-128-euclidean",
		"example/data/nearest-neighbors-datasets", 100, -1, 5)
}
