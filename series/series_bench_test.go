package series

import "testing"

func BenchmarkSeriesMap1M(b *testing.B) {
	vals := make([]float64, 1_000_000)
	s := New("x", vals)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Map(func(v float64) float64 { return v * 2 })
	}
}

func BenchmarkSeriesReduce1M(b *testing.B) {
	vals := make([]float64, 1_000_000)
	s := New("x", vals)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Reduce(0, func(acc, v float64) float64 { return acc + v })
	}
}
