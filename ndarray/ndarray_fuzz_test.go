package ndarray

import (
	"math"
	"testing"
)

func FuzzReshapeRoundTrip(f *testing.F) {
	f.Add(2, 3)
	f.Add(4, 5)
	f.Fuzz(func(t *testing.T, r, c int) {
		if r <= 0 || c <= 0 || r > 64 || c > 64 {
			t.Skip()
		}
		size := r * c
		data := make([]float64, size)
		for i := range data {
			data[i] = float64(i%17) - 8
		}
		a, err := FromSlice(data, r, c)
		if err != nil {
			t.Fatalf("FromSlice failed: %v", err)
		}
		flat := a.Flatten()
		back, err := flat.Reshape(r, c)
		if err != nil {
			t.Fatalf("reshape failed: %v", err)
		}
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				if a.At(i, j) != back.At(i, j) {
					t.Fatalf("mismatch at (%d,%d): %v vs %v", i, j, a.At(i, j), back.At(i, j))
				}
			}
		}
	})
}

func FuzzTransposeInvolution(f *testing.F) {
	f.Add(2, 2)
	f.Add(3, 4)
	f.Fuzz(func(t *testing.T, r, c int) {
		if r <= 0 || c <= 0 || r > 32 || c > 32 {
			t.Skip()
		}
		data := make([]float64, r*c)
		for i := range data {
			data[i] = float64((i % 11) - 5)
		}
		a, err := FromSlice(data, r, c)
		if err != nil {
			t.Fatalf("FromSlice failed: %v", err)
		}
		tt := a.T().T()
		if tt.Shape()[0] != r || tt.Shape()[1] != c {
			t.Fatalf("shape mismatch after transpose twice: got %v want [%d %d]", tt.Shape(), r, c)
		}
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				if math.Abs(tt.At(i, j)-a.At(i, j)) > 1e-12 {
					t.Fatalf("value mismatch at (%d,%d): %v vs %v", i, j, tt.At(i, j), a.At(i, j))
				}
			}
		}
	})
}
