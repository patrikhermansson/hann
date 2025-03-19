package pqivf_test

import (
	"os"
	"sync"
	"testing"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/pqivf"
)

func TestPQIVF_BasicOperations(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers)

	// Test Add.
	vec1 := []float32{1, 2, 3, 4, 5, 6}
	if err := idx.Add(1, vec1); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	stats := idx.Stats()
	if stats.Count != 1 {
		t.Errorf("expected count 1, got %d", stats.Count)
	}

	// Test Update.
	vec1upd := []float32{6, 5, 4, 3, 2, 1}
	if err := idx.Update(1, vec1upd); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Test Delete.
	if err := idx.Delete(1); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	stats = idx.Stats()
	if stats.Count != 0 {
		t.Errorf("expected count 0 after delete, got %d", stats.Count)
	}
}

func TestPQIVF_Search(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers)

	// Insert several vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
	}
	for id, vec := range vectors {
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}

	query := []float32{1, 2, 3, 4, 5, 6}
	neighbors, err := idx.Search(query, 2, core.Euclidean)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors, got %d", len(neighbors))
	}
	// If an exact match exists, expect id 1 to be the closest.
	if neighbors[0].ID != 1 {
		t.Errorf("expected neighbor id 1 as closest, got %d", neighbors[0].ID)
	}
}

func TestPQIVF_RangeSearch(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers)

	// Insert vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {1, 2, 3, 4, 5, 7},
		3: {10, 10, 10, 10, 10, 10},
	}
	for id, vec := range vectors {
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}

	query := []float32{1, 2, 3, 4, 5, 6}
	ids, err := idx.RangeSearch(query, 2.0, core.Euclidean)
	if err != nil {
		t.Fatalf("RangeSearch failed: %v", err)
	}
	if len(ids) == 0 {
		t.Errorf("expected at least one id in range, got 0")
	}
}

func TestPQIVF_SaveLoad(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers)

	// Insert a couple of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
	}
	for id, vec := range vectors {
		if err := idx.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}

	filePath := "test_pqivf.gob"
	defer os.Remove(filePath)

	if err := idx.Save(filePath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	newIdx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers)
	if err := newIdx.Load(filePath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	stats := newIdx.Stats()
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after load, got %d", len(vectors), stats.Count)
	}
}

func TestPQIVF_ConcurrentOperations(t *testing.T) {
	dim := 6
	coarseK := 3
	numSubquantizers := 2
	idx := pqivf.NewPQIVFIndex(dim, coarseK, numSubquantizers)
	numVectors := 1000
	var wg sync.WaitGroup

	for i := 0; i < numVectors; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			vec := []float32{
				float32(id),
				float32(id + 1),
				float32(id + 2),
				float32(id + 3),
				float32(id + 4),
				float32(id + 5),
			}
			if err := idx.Add(id, vec); err != nil {
				t.Errorf("Add failed for id %d: %v", id, err)
			}
		}(i)
	}
	wg.Wait()

	stats := idx.Stats()
	if stats.Count != numVectors {
		t.Errorf("expected %d vectors, got %d", numVectors, stats.Count)
	}
}
