package ndarray

import (
	"math"
	"testing"
)

func TestAddAndMatMul(t *testing.T) {
	a, _ := FromSlice([]float64{1, 2, 3, 4}, 2, 2)
	b, _ := FromSlice([]float64{5, 6, 7, 8}, 2, 2)

	sum, err := Add(a, b)
	if err != nil {
		t.Fatalf("Add error: %v", err)
	}
	if got := sum.At(1, 1); got != 12 {
		t.Fatalf("sum(1,1)=%v want 12", got)
	}

	mm, err := MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul error: %v", err)
	}
	if got := mm.At(0, 0); got != 19 {
		t.Fatalf("matmul(0,0)=%v want 19", got)
	}
}

func TestReshapeAndTranspose(t *testing.T) {
	a, _ := FromSlice([]int64{1, 2, 3, 4, 5, 6}, 2, 3)
	r, err := a.Reshape(3, 2)
	if err != nil {
		t.Fatalf("reshape error: %v", err)
	}
	if r.At(2, 1) != 6 {
		t.Fatalf("reshape value mismatch")
	}
	tr := a.T()
	if tr.Shape()[0] != 3 || tr.Shape()[1] != 2 {
		t.Fatalf("transpose shape mismatch: %v", tr.Shape())
	}
	if tr.At(1, 0) != 2 {
		t.Fatalf("transpose value mismatch")
	}
}

func TestSolveStable(t *testing.T) {
	a, _ := FromSlice([]float64{
		3, 1,
		1, 2,
	}, 2, 2)
	b, _ := FromSlice([]float64{9, 8}, 2)
	x, err := Solve(a, b)
	if err != nil {
		t.Fatalf("Solve error: %v", err)
	}
	if got := x.At(0); math.Abs(got-2) > 1e-9 {
		t.Fatalf("x0=%v want 2", got)
	}
	if got := x.At(1); math.Abs(got-3) > 1e-9 {
		t.Fatalf("x1=%v want 3", got)
	}
}

func TestSVDReconstruction(t *testing.T) {
	a, _ := FromSlice([]float64{
		3, 1,
		0, 2,
	}, 2, 2)
	u, s, vt, err := SVD(a)
	if err != nil {
		t.Fatalf("SVD error: %v", err)
	}
	if u == nil || s == nil || vt == nil {
		t.Fatalf("SVD returned nil component")
	}
	sdiag := zeros2D(2, 2)
	sdiag.Set(s.At(0), 0, 0)
	sdiag.Set(s.At(1), 1, 1)
	us, err := MatMul(u, sdiag)
	if err != nil {
		t.Fatalf("MatMul(U,S) error: %v", err)
	}
	recon, err := MatMul(us, vt)
	if err != nil {
		t.Fatalf("MatMul(US,Vt) error: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(recon.At(i, j)-a.At(i, j)) > 1e-5 {
				t.Fatalf("reconstruction mismatch at (%d,%d): got %v want %v", i, j, recon.At(i, j), a.At(i, j))
			}
		}
	}
}

func zeros2D(r, c int) *NDArray[float64] {
	return &NDArray[float64]{
		data:    make([]float64, r*c),
		shape:   []int{r, c},
		strides: []int{c, 1},
	}
}
