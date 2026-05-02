package io

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/kalpit-sharma-dev/gods/dataframe"
)

func TestCSVReadWriteRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "data.csv")

	df, err := dataframe.FromMap(map[string]any{
		"id":   []int64{1, 2, 3},
		"name": []string{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("FromMap failed: %v", err)
	}
	if err := WriteCSV(df, path); err != nil {
		t.Fatalf("WriteCSV failed: %v", err)
	}
	got, err := ReadCSV(path)
	if err != nil {
		t.Fatalf("ReadCSV failed: %v", err)
	}
	rows, cols := got.Shape()
	if rows != 3 || cols != 2 {
		t.Fatalf("unexpected shape: got (%d,%d)", rows, cols)
	}
}

func TestJSONReadWriteRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "data.json")

	df, err := dataframe.FromMap(map[string]any{
		"id":   []int64{1, 2},
		"name": []string{"x", "y"},
	})
	if err != nil {
		t.Fatalf("FromMap failed: %v", err)
	}

	if err := WriteJSON(df, path, JSONOrientRecords); err != nil {
		t.Fatalf("WriteJSON failed: %v", err)
	}
	got, err := ReadJSON(path)
	if err != nil {
		t.Fatalf("ReadJSON failed: %v", err)
	}
	rows, cols := got.Shape()
	if rows != 2 || cols != 2 {
		t.Fatalf("unexpected shape: got (%d,%d)", rows, cols)
	}
}

func TestReadCSVChunked(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "chunk.csv")
	content := "id,name\n1,a\n2,b\n3,c\n4,d\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture failed: %v", err)
	}
	opts := DefaultCSVReadOptions()
	opts.ChunkSize = 2
	ch, errCh := ReadCSVChunked(path, opts)
	chunks := 0
	for range ch {
		chunks++
	}
	if err := <-errCh; err != nil {
		t.Fatalf("ReadCSVChunked failed: %v", err)
	}
	if chunks != 2 {
		t.Fatalf("expected 2 chunks, got %d", chunks)
	}
}

func TestReadJSONColumnsOrientation(t *testing.T) {
	input := map[string][]any{
		"id":   {float64(1), float64(2)},
		"name": {"a", "b"},
	}
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	df, err := ReadJSONBytes(raw, JSONReadOptions{Orientation: JSONOrientColumns})
	if err != nil {
		t.Fatalf("ReadJSONBytes failed: %v", err)
	}
	rows, cols := df.Shape()
	if rows != 2 || cols != 2 {
		t.Fatalf("unexpected shape: got (%d,%d)", rows, cols)
	}
}

func TestCSVReadBoundaryNumerics(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "numbers.csv")
	content := "x,y,z\n9223372036854775807,NaN,Inf\n-9223372036854775808,-Inf,-0.0\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture failed: %v", err)
	}
	df, err := ReadCSV(path)
	if err != nil {
		t.Fatalf("ReadCSV failed: %v", err)
	}
	rows, cols := df.Shape()
	if rows != 2 || cols != 3 {
		t.Fatalf("unexpected shape: got (%d,%d)", rows, cols)
	}
	row0, err := df.Row(0)
	if err != nil {
		t.Fatalf("row error: %v", err)
	}
	if _, ok := row0["x"].(int64); !ok {
		t.Fatalf("expected x to infer as int64, got %T", row0["x"])
	}
	// Default null values include "NaN", so y should be null by default.
	if row0["y"] != nil {
		t.Fatalf("expected y to be null under default null-values handling, got %T (%v)", row0["y"], row0["y"])
	}
	// Validate boundary float parsing behavior with NaN excluded from null set.
	opts := DefaultCSVReadOptions()
	opts.NullValues = []string{"", "NA", "null"}
	df2, err := ReadCSV(path, opts)
	if err != nil {
		t.Fatalf("ReadCSV with custom null values failed: %v", err)
	}
	row0b, err := df2.Row(0)
	if err != nil {
		t.Fatalf("row error: %v", err)
	}
	if y, ok := row0b["y"].(float64); !ok || !math.IsNaN(y) {
		t.Fatalf("expected y to be NaN float64 with custom null values, got %T (%v)", row0b["y"], row0b["y"])
	}
}
