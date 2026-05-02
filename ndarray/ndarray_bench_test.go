package ndarray

import "testing"

func BenchmarkNDArrayAdd(b *testing.B) {
	size := 1_000_000
	a := Full(1.0, size)
	c := Full(2.0, size)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Add(a, c)
	}
}

func BenchmarkNDArrayMatMul(b *testing.B) {
	n := 256
	a := Full(1.0, n, n)
	c := Full(2.0, n, n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatMul(a, c)
	}
}
