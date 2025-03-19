package hnsw

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/habedi/hann/core"
)

// Node represents an element in the HNSW graph.
type Node struct {
	ID     int
	Vector []float32
	Level  int
	// Links maps each level to a slice of neighbor nodes.
	Links map[int][]*Node
}

// HNSWIndex implements a simplified HNSW index.
type HNSWIndex struct {
	mu         sync.RWMutex
	dimension  int
	entryPoint *Node
	maxLevel   int
	nodes      map[int]*Node

	// Parameters (simplified)
	M  int // maximum number of neighbors per node
	ef int // search parameter (candidate list size for search)
}

func init() {
	rand.Seed(time.Now().UnixNano())
}

// NewHNSW creates a new HNSW index with the given dimension and parameters.
func NewHNSW(dimension int, M int, ef int) *HNSWIndex {
	return &HNSWIndex{
		dimension: dimension,
		nodes:     make(map[int]*Node),
		maxLevel:  -1,
		M:         M,
		ef:        ef,
	}
}

// randomLevel generates a random level for a new node using an exponential distribution.
func (h *HNSWIndex) randomLevel() int {
	if h.M <= 1 {
		return 0
	}
	// Use negative logarithm to sample from an exponential distribution.
	level := int(-math.Log(rand.Float64()) / math.Log(float64(h.M)))
	return level
}

// euclidean is a simple Euclidean distance function.
func euclidean(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// Add inserts a vector with a given id into the HNSW index.
func (h *HNSWIndex) Add(id int, vector []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if len(vector) != h.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), h.dimension)
	}
	if _, exists := h.nodes[id]; exists {
		return fmt.Errorf("id %d already exists", id)
	}

	level := h.randomLevel()
	newNode := &Node{
		ID:     id,
		Vector: vector,
		Level:  level,
		Links:  make(map[int][]*Node),
	}
	h.nodes[id] = newNode

	// If index is empty, set new node as entry point.
	if h.entryPoint == nil {
		h.entryPoint = newNode
		h.maxLevel = level
		return nil
	}

	// If the new node has a higher level than the current max, update the entry point.
	if level > h.maxLevel {
		h.entryPoint = newNode
		h.maxLevel = level
	}

	// Improved linking: For each level up to newNode.Level, connect to at most M nearest neighbors.
	for l := 0; l <= newNode.Level; l++ {
		type candidateLink struct {
			node *Node
			dist float64
		}
		var candidates []candidateLink
		// Consider only nodes that have level >= l.
		for _, other := range h.nodes {
			if other.ID == newNode.ID {
				continue
			}
			if other.Level >= l {
				d := euclidean(newNode.Vector, other.Vector)
				candidates = append(candidates, candidateLink{node: other, dist: d})
			}
		}
		// Sort candidates by increasing distance.
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].dist < candidates[j].dist
		})
		// Select up to M nearest neighbors.
		num := len(candidates)
		if num > h.M {
			num = h.M
		}
		var selected []*Node
		for i := 0; i < num; i++ {
			selected = append(selected, candidates[i].node)
		}
		newNode.Links[l] = selected

		// For each selected neighbor, add newNode to neighbor's Links at level l, then trim if necessary.
		for _, neighbor := range selected {
			neighbor.Links[l] = append(neighbor.Links[l], newNode)
			if len(neighbor.Links[l]) > h.M {
				sort.Slice(neighbor.Links[l], func(i, j int) bool {
					return euclidean(neighbor.Vector, neighbor.Links[l][i].Vector) < euclidean(neighbor.Vector, neighbor.Links[l][j].Vector)
				})
				neighbor.Links[l] = neighbor.Links[l][:h.M]
			}
		}
	}

	return nil
}

// Delete removes the vector with the given id from the index.
// This simple implementation removes the node and cleans up links.
func (h *HNSWIndex) Delete(id int) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	_, exists := h.nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	// Remove the node from all neighbor links.
	for _, n := range h.nodes {
		for l, neighbors := range n.Links {
			newNeighbors := make([]*Node, 0, len(neighbors))
			for _, neighbor := range neighbors {
				if neighbor.ID != id {
					newNeighbors = append(newNeighbors, neighbor)
				}
			}
			n.Links[l] = newNeighbors
		}
	}
	delete(h.nodes, id)

	// Update entry point if necessary.
	if h.entryPoint != nil && h.entryPoint.ID == id {
		h.entryPoint = nil
		for _, n := range h.nodes {
			h.entryPoint = n
			break
		}
	}
	return nil
}

// Update modifies the vector associated with the given id.
// (For simplicity, it updates the vector without re-linking.)
func (h *HNSWIndex) Update(id int, vector []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	node, exists := h.nodes[id]
	if !exists {
		return fmt.Errorf("id %d not found", id)
	}
	if len(vector) != h.dimension {
		return fmt.Errorf("vector dimension %d does not match index dimension %d", len(vector), h.dimension)
	}
	node.Vector = vector
	return nil
}

// Search returns the ids and distances of the k nearest neighbors for a query vector.
// The search performs a greedy descent from the highest level down and then a best-first
// search at level 0 using the ef parameter.
func (h *HNSWIndex) Search(query []float32, k int, distance core.DistanceFunc) ([]core.Neighbor, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(query) != h.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), h.dimension)
	}
	if h.entryPoint == nil {
		return nil, errors.New("index is empty")
	}

	// Greedy search from the top level down to level 1.
	currNode := h.entryPoint
	for level := h.maxLevel; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			for _, neighbor := range currNode.Links[level] {
				if distance(query, neighbor.Vector) < distance(query, currNode.Vector) {
					currNode = neighbor
					changed = true
				}
			}
		}
	}

	// Best-first search at level 0 using candidate list of size ef.
	type candidate struct {
		node *Node
		dist float64
	}
	candidates := []candidate{{node: currNode, dist: distance(query, currNode.Vector)}}
	visited := make(map[int]bool)
	visited[currNode.ID] = true

	changed := true
	for changed {
		changed = false
		for i := 0; i < len(candidates); i++ {
			cand := candidates[i]
			for _, neighbor := range cand.node.Links[0] {
				if visited[neighbor.ID] {
					continue
				}
				visited[neighbor.ID] = true
				d := distance(query, neighbor.Vector)
				// If the candidate list is not full or this neighbor is closer than the farthest candidate.
				if len(candidates) < h.ef || d < candidates[len(candidates)-1].dist {
					candidates = append(candidates, candidate{node: neighbor, dist: d})
					sort.Slice(candidates, func(i, j int) bool {
						return candidates[i].dist < candidates[j].dist
					})
					if len(candidates) > h.ef {
						candidates = candidates[:h.ef]
					}
					changed = true
				}
			}
		}
	}

	// Sort the final candidates and return the top k.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})
	if k > len(candidates) {
		k = len(candidates)
	}
	results := make([]core.Neighbor, k)
	for i := 0; i < k; i++ {
		results[i] = core.Neighbor{ID: candidates[i].node.ID, Distance: candidates[i].dist}
	}
	return results, nil
}

// RangeSearch returns all neighbor ids within the specified radius.
func (h *HNSWIndex) RangeSearch(query []float32, radius float64, distance core.DistanceFunc) ([]int, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if len(query) != h.dimension {
		return nil, fmt.Errorf("query dimension %d does not match index dimension %d", len(query), h.dimension)
	}
	var ids []int
	for _, node := range h.nodes {
		if d := distance(query, node.Vector); d <= radius {
			ids = append(ids, node.ID)
		}
	}
	return ids, nil
}

// Stats returns metadata about the index.
func (h *HNSWIndex) Stats() core.IndexStats {
	h.mu.RLock()
	defer h.mu.RUnlock()
	count := len(h.nodes)
	// Rough estimation: each vector is dimension*4 bytes.
	size := count * h.dimension * 4
	return core.IndexStats{
		Count:     count,
		Dimension: h.dimension,
		Size:      size,
	}
}

// Save persists the index state to the specified file.
func (h *HNSWIndex) Save(path string) error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	return enc.Encode(h)
}

// Load initializes the index from a previously saved state.
func (h *HNSWIndex) Load(path string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	return dec.Decode(h)
}

// --- Custom Gob Serialization ---
//
// Because Node.Links is cyclic (nodes reference each other),
// we serialize the index into an intermediate structure that stores neighbor IDs only.

type serializedNode struct {
	ID     int
	Vector []float32
	Level  int
	Links  map[int][]int // For each level, list of neighbor IDs.
}

type serializedHNSW struct {
	Dimension  int
	EntryPoint int // entry point ID; -1 if nil.
	MaxLevel   int
	Nodes      map[int]serializedNode
}

func (h *HNSWIndex) GobEncode() ([]byte, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	sh := serializedHNSW{
		Dimension:  h.dimension,
		MaxLevel:   h.maxLevel,
		Nodes:      make(map[int]serializedNode, len(h.nodes)),
		EntryPoint: -1,
	}
	if h.entryPoint != nil {
		sh.EntryPoint = h.entryPoint.ID
	}
	for id, node := range h.nodes {
		sn := serializedNode{
			ID:     node.ID,
			Vector: node.Vector,
			Level:  node.Level,
			Links:  make(map[int][]int),
		}
		for level, neighbors := range node.Links {
			for _, neighbor := range neighbors {
				sn.Links[level] = append(sn.Links[level], neighbor.ID)
			}
		}
		sh.Nodes[id] = sn
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(sh); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (h *HNSWIndex) GobDecode(data []byte) error {
	var sh serializedHNSW
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)
	if err := dec.Decode(&sh); err != nil {
		return err
	}
	nodes := make(map[int]*Node, len(sh.Nodes))
	// First pass: create nodes without links.
	for id, sn := range sh.Nodes {
		nodes[id] = &Node{
			ID:     sn.ID,
			Vector: sn.Vector,
			Level:  sn.Level,
			Links:  make(map[int][]*Node),
		}
	}
	// Second pass: reconstruct links from neighbor IDs.
	for id, sn := range sh.Nodes {
		currNode := nodes[id]
		for level, neighborIDs := range sn.Links {
			for _, nid := range neighborIDs {
				if neighbor, ok := nodes[nid]; ok {
					currNode.Links[level] = append(currNode.Links[level], neighbor)
				}
			}
		}
	}
	h.dimension = sh.Dimension
	h.maxLevel = sh.MaxLevel
	h.nodes = nodes
	if sh.EntryPoint >= 0 {
		h.entryPoint = nodes[sh.EntryPoint]
	} else {
		h.entryPoint = nil
	}
	return nil
}

// Ensure HNSWIndex implements the core.Index interface.
var _ core.Index = (*HNSWIndex)(nil)

func init() {
	gob.Register(&HNSWIndex{})
	gob.Register(&Node{})
}
