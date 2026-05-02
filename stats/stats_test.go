package stats

import (
	"math"
	"testing"
)

func TestDescriptiveStats(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	if got := Mean(data); math.Abs(got-3) > 1e-12 {
		t.Fatalf("Mean mismatch: got %v", got)
	}
	if got := Median(data); math.Abs(got-3) > 1e-12 {
		t.Fatalf("Median mismatch: got %v", got)
	}
	if got := Variance(data); math.Abs(got-2.5) > 1e-12 {
		t.Fatalf("Variance mismatch: got %v", got)
	}
}

func TestDistributions(t *testing.T) {
	n := Normal(0, 1)
	pdf0 := n.PDF(0)
	if pdf0 <= 0 {
		t.Fatalf("normal pdf at 0 should be > 0")
	}
	p := n.CDF(0)
	if p <= 0 || p >= 1 {
		t.Fatalf("normal cdf at 0 should be in (0,1), got %v", p)
	}

	u := Uniform(0, 10)
	if u.PDF(-1) != 0 {
		t.Fatalf("uniform pdf outside support should be 0")
	}
}
