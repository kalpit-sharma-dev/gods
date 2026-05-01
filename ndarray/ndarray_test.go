package ndarray

import "testing"

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
