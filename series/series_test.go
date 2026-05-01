package series

import "testing"

func TestSeriesBasics(t *testing.T) {
	s := New("x", []int64{1, 2, 3})
	if s.Len() != 3 {
		t.Fatalf("expected len 3, got %d", s.Len())
	}
	if s.NullCount() != 0 {
		t.Fatalf("expected no nulls")
	}
	s.SetNull(1)
	if s.NullCount() != 1 {
		t.Fatalf("expected 1 null, got %d", s.NullCount())
	}
	v, ok := s.At(0)
	if !ok || v != 1 {
		t.Fatalf("unexpected first value: %v %v", v, ok)
	}
}

func TestSeriesMapAndStats(t *testing.T) {
	s := New("x", []float64{1, 2, 3, 4})
	out := s.Map(func(v float64) float64 { return v * 2 })
	if v, _ := out.At(2); v != 6 {
		t.Fatalf("expected mapped value 6, got %v", v)
	}
	m, err := Mean(out)
	if err != nil {
		t.Fatalf("mean error: %v", err)
	}
	if m != 5 {
		t.Fatalf("expected mean 5, got %v", m)
	}
}

func TestSeriesFillAndDropNull(t *testing.T) {
	s := WithNulls("x", []int64{1, 2, 3}, []bool{false, true, false})
	filled := s.FillNa(9)
	if v, _ := filled.At(1); v != 9 {
		t.Fatalf("expected filled value 9, got %d", v)
	}
	dropped := s.DropNa()
	if dropped.Len() != 2 {
		t.Fatalf("expected len 2 after dropna, got %d", dropped.Len())
	}
}
