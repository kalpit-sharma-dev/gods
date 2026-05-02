package bitmap

import "math/bits"

// Bitmap is a compact validity mask. Bit i = 1 means valid, 0 means null.
type Bitmap struct {
	data []uint64
	len  int
}

// New creates a bitmap with n bits initialized to valid (1).
func New(n int) *Bitmap {
	if n < 0 {
		n = 0
	}
	words := (n + 63) / 64
	data := make([]uint64, words)
	for i := range data {
		data[i] = ^uint64(0)
	}
	if n > 0 && n%64 != 0 {
		rem := uint(n % 64)
		data[words-1] = (uint64(1) << rem) - 1
	}
	return &Bitmap{data: data, len: n}
}

// Set sets the validity bit at index i.
func (b *Bitmap) Set(i int, valid bool) {
	if b == nil || i < 0 || i >= b.len {
		return
	}
	word := i / 64
	bit := uint(i % 64)
	mask := uint64(1) << bit
	if valid {
		b.data[word] |= mask
		return
	}
	b.data[word] &^= mask
}

// IsValid reports whether index i is valid.
func (b *Bitmap) IsValid(i int) bool {
	if b == nil || i < 0 || i >= b.len {
		return false
	}
	word := i / 64
	bit := uint(i % 64)
	return (b.data[word] & (uint64(1) << bit)) != 0
}

// IsNull reports whether index i is null.
func (b *Bitmap) IsNull(i int) bool {
	return !b.IsValid(i)
}

// ValidCount returns the number of valid bits.
func (b *Bitmap) ValidCount() int {
	if b == nil {
		return 0
	}
	total := 0
	for i, w := range b.data {
		if i == len(b.data)-1 && b.len%64 != 0 {
			rem := uint(b.len % 64)
			w &= (uint64(1) << rem) - 1
		}
		total += bits.OnesCount64(w)
	}
	return total
}

// NullCount returns the number of null bits.
func (b *Bitmap) NullCount() int {
	if b == nil {
		return 0
	}
	return b.len - b.ValidCount()
}

// Clone returns a deep copy of the bitmap.
func (b *Bitmap) Clone() *Bitmap {
	if b == nil {
		return nil
	}
	dup := make([]uint64, len(b.data))
	copy(dup, b.data)
	return &Bitmap{data: dup, len: b.len}
}
