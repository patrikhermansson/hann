package hnsw_test

import (
	"github.com/habedi/hann/core"
	"os"
	"sync"
	"testing"

	"github.com/habedi/hann/hnsw"
)

func TestHNSWIndex_BasicOperations(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10)

	// Arrange: create a 6D vector.
	vec1 := []float32{1, 2, 3, 4, 5, 6}

	// Act & Assert: Test Add.
	if err := index.Add(1, vec1); err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	stats := index.Stats()
	if stats.Count != 1 {
		t.Errorf("expected count 1, got %d", stats.Count)
	}

	// Act & Assert: Test Update.
	vec1Upd := []float32{6, 5, 4, 3, 2, 1}
	if err := index.Update(1, vec1Upd); err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	// Act & Assert: Test Delete.
	if err := index.Delete(1); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	stats = index.Stats()
	if stats.Count != 0 {
		t.Errorf("expected count 0 after delete, got %d", stats.Count)
	}
}

func TestHNSWIndex_Search(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10)
	// Arrange: insert multiple vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
		3: {1, 1, 1, 1, 1, 1},
		4: {2, 2, 2, 2, 2, 2},
	}
	for id, vec := range vectors {
		if err := index.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}

	query := []float32{1, 2, 3, 4, 5, 6}

	// Act: search for 2 nearest neighbors.
	neighbors, err := index.Search(query, 2, core.Euclidean)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Assert: verify that we got 2 neighbors and they are sorted by distance.
	if len(neighbors) != 2 {
		t.Errorf("expected 2 neighbors, got %d", len(neighbors))
	}
	if neighbors[0].Distance > neighbors[1].Distance {
		t.Errorf("neighbors not sorted by distance")
	}
}

func TestHNSWIndex_RangeSearch(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10)
	// Arrange: insert some vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {2, 3, 4, 5, 6, 7},
		3: {10, 10, 10, 10, 10, 10},
	}
	for id, vec := range vectors {
		if err := index.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}
	query := []float32{1, 2, 3, 4, 5, 6}

	// Act: perform a range search with a radius that should capture two vectors.
	ids, err := index.RangeSearch(query, 5.0, core.Euclidean)
	if err != nil {
		t.Fatalf("RangeSearch failed: %v", err)
	}

	// Assert: Expect at least 2 vector ids within the given radius.
	if len(ids) < 2 {
		t.Errorf("expected at least 2 ids within radius, got %d", len(ids))
	}
}

func TestHNSWIndex_SaveLoad(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10)
	// Arrange: add a couple of vectors.
	vectors := map[int][]float32{
		1: {1, 2, 3, 4, 5, 6},
		2: {6, 5, 4, 3, 2, 1},
	}
	for id, vec := range vectors {
		if err := index.Add(id, vec); err != nil {
			t.Fatalf("Add failed for id %d: %v", id, err)
		}
	}
	filePath := "test_hnsw.gob"
	defer os.Remove(filePath)

	// Act: Save the index.
	if err := index.Save(filePath); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Act: Create a new index and load the saved state.
	newIndex := hnsw.NewHNSW(dim, 5, 10)
	if err := newIndex.Load(filePath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	stats := newIndex.Stats()
	// Assert: The count should match the original.
	if stats.Count != len(vectors) {
		t.Errorf("expected count %d after load, got %d", len(vectors), stats.Count)
	}
}

func TestHNSWIndex_ConcurrentOperations(t *testing.T) {
	dim := 6
	index := hnsw.NewHNSW(dim, 5, 10)
	numVectors := 1000
	var wg sync.WaitGroup

	// Arrange & Act: Concurrently add many vectors.
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
			if err := index.Add(id, vec); err != nil {
				t.Errorf("Add failed for id %d: %v", id, err)
			}
		}(i)
	}
	wg.Wait()

	stats := index.Stats()
	if stats.Count != numVectors {
		t.Errorf("expected %d vectors, got %d", numVectors, stats.Count)
	}
}
