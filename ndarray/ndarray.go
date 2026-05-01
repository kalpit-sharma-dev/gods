package ndarray

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

// NDArray is a typed N-dimensional array in row-major order.
type NDArray[T any] struct {
	data    []T
	shape   []int
	strides []int
}

// SliceSpec describes a single-axis slice [Start:Stop:Step].
type SliceSpec struct{ Start, Stop, Step *int }

// Number captures integer and float numeric types supported by NDArray operations.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// Zeros creates an array of zeros with the provided shape.
func Zeros[T Number](shape ...int) *NDArray[T] {
	size := product(shape)
	return &NDArray[T]{
		data:    make([]T, size),
		shape:   append([]int(nil), shape...),
		strides: computeStrides(shape),
	}
}

// Ones creates an array of ones with the provided shape.
func Ones[T Number](shape ...int) *NDArray[T] {
	arr := Zeros[T](shape...)
	var one T = 1
	for i := range arr.data {
		arr.data[i] = one
	}
	return arr
}

// Full creates an array filled with a scalar value.
func Full[T Number](value T, shape ...int) *NDArray[T] {
	arr := Zeros[T](shape...)
	for i := range arr.data {
		arr.data[i] = value
	}
	return arr
}

// FromSlice creates an NDArray from flat data and shape.
func FromSlice[T Number](data []T, shape ...int) (*NDArray[T], error) {
	if product(shape) != len(data) {
		return nil, fmt.Errorf("gods/ndarray: data length %d does not match shape size %d", len(data), product(shape))
	}
	out := make([]T, len(data))
	copy(out, data)
	return &NDArray[T]{
		data:    out,
		shape:   append([]int(nil), shape...),
		strides: computeStrides(shape),
	}, nil
}

// Arange creates a 1D array of evenly spaced values [start, stop) by step.
func Arange[T Number](start, stop, step T) *NDArray[T] {
	if step == 0 {
		return &NDArray[T]{shape: []int{0}, strides: []int{1}}
	}
	values := make([]T, 0)
	for v := start; ; v += step {
		if step > 0 && v >= stop {
			break
		}
		if step < 0 && v <= stop {
			break
		}
		values = append(values, v)
	}
	return &NDArray[T]{
		data:    values,
		shape:   []int{len(values)},
		strides: []int{1},
	}
}

// Linspace creates n evenly spaced values from start to stop inclusive.
func Linspace(start, stop float64, n int) *NDArray[float64] {
	if n <= 0 {
		return &NDArray[float64]{shape: []int{0}, strides: []int{1}}
	}
	if n == 1 {
		return &NDArray[float64]{data: []float64{start}, shape: []int{1}, strides: []int{1}}
	}
	step := (stop - start) / float64(n-1)
	data := make([]float64, n)
	for i := range data {
		data[i] = start + float64(i)*step
	}
	data[n-1] = stop
	return &NDArray[float64]{data: data, shape: []int{n}, strides: []int{1}}
}

// Random creates a float64 array with uniform random values in [0, 1).
func Random(shape ...int) *NDArray[float64] {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	size := product(shape)
	data := make([]float64, size)
	for i := range data {
		data[i] = r.Float64()
	}
	return &NDArray[float64]{data: data, shape: append([]int(nil), shape...), strides: computeStrides(shape)}
}

// RandomNormal creates a float64 array with normal random values.
func RandomNormal(mean, std float64, shape ...int) *NDArray[float64] {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	size := product(shape)
	data := make([]float64, size)
	for i := range data {
		data[i] = mean + std*r.NormFloat64()
	}
	return &NDArray[float64]{data: data, shape: append([]int(nil), shape...), strides: computeStrides(shape)}
}

// Shape returns a copy of the shape.
func (a *NDArray[T]) Shape() []int {
	if a == nil {
		return nil
	}
	return append([]int(nil), a.shape...)
}

// Ndim returns number of dimensions.
func (a *NDArray[T]) Ndim() int {
	if a == nil {
		return 0
	}
	return len(a.shape)
}

// Size returns total element count.
func (a *NDArray[T]) Size() int {
	if a == nil {
		return 0
	}
	return len(a.data)
}

// Strides returns a copy of strides.
func (a *NDArray[T]) Strides() []int {
	if a == nil {
		return nil
	}
	return append([]int(nil), a.strides...)
}

// String returns a compact representation.
func (a *NDArray[T]) String() string {
	if a == nil {
		return "NDArray<nil>"
	}
	const max = 12
	var b strings.Builder
	fmt.Fprintf(&b, "NDArray(shape=%v, size=%d, data=[", a.shape, len(a.data))
	limit := len(a.data)
	if limit > max {
		limit = max
	}
	for i := 0; i < limit; i++ {
		if i > 0 {
			b.WriteString(" ")
		}
		fmt.Fprintf(&b, "%v", a.data[i])
	}
	if len(a.data) > limit {
		b.WriteString(" ...")
	}
	b.WriteString("])")
	return b.String()
}

// At returns element at index.
func (a *NDArray[T]) At(idx ...int) T {
	var zero T
	if a == nil {
		return zero
	}
	flat := a.flatIndexOrPanic(idx...)
	return a.data[flat]
}

// Set sets element at index.
func (a *NDArray[T]) Set(val T, idx ...int) {
	if a == nil {
		return
	}
	flat := a.flatIndexOrPanic(idx...)
	a.data[flat] = val
}

// Slice creates a sliced copy of the array.
func (a *NDArray[T]) Slice(specs ...SliceSpec) (*NDArray[T], error) {
	if a == nil {
		return nil, fmt.Errorf("gods/ndarray: nil array")
	}
	if len(specs) > len(a.shape) {
		return nil, fmt.Errorf("gods/ndarray: %d slice specs exceed ndim %d", len(specs), len(a.shape))
	}
	for len(specs) < len(a.shape) {
		specs = append(specs, SliceSpec{})
	}
	ranges := make([][]int, len(a.shape))
	newShape := make([]int, len(a.shape))
	for axis := range a.shape {
		start, stop, step, err := normalizeSliceSpec(specs[axis], a.shape[axis])
		if err != nil {
			return nil, fmt.Errorf("gods/ndarray: axis %d: %w", axis, err)
		}
		values := make([]int, 0)
		if step > 0 {
			for i := start; i < stop; i += step {
				values = append(values, i)
			}
		} else {
			for i := start; i > stop; i += step {
				values = append(values, i)
			}
		}
		ranges[axis] = values
		newShape[axis] = len(values)
	}
	outSize := product(newShape)
	out := make([]T, 0, outSize)
	indices := make([]int, len(a.shape))
	var walk func(int)
	walk = func(axis int) {
		if axis == len(a.shape) {
			out = append(out, a.At(indices...))
			return
		}
		for _, i := range ranges[axis] {
			indices[axis] = i
			walk(axis + 1)
		}
	}
	walk(0)
	return &NDArray[T]{data: out, shape: newShape, strides: computeStrides(newShape)}, nil
}

// Flatten returns a 1D copy.
func (a *NDArray[T]) Flatten() *NDArray[T] {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	copy(out, a.data)
	return &NDArray[T]{data: out, shape: []int{len(out)}, strides: []int{1}}
}

// Reshape returns a view with new shape when total size matches.
func (a *NDArray[T]) Reshape(newShape ...int) (*NDArray[T], error) {
	if a == nil {
		return nil, fmt.Errorf("gods/ndarray: nil array")
	}
	if product(newShape) != len(a.data) {
		return nil, fmt.Errorf("gods/ndarray: cannot reshape size %d into shape %v", len(a.data), newShape)
	}
	return &NDArray[T]{
		data:    a.data,
		shape:   append([]int(nil), newShape...),
		strides: computeStrides(newShape),
	}, nil
}

// T returns transpose by reversing axes.
func (a *NDArray[T]) T() *NDArray[T] {
	if a == nil {
		return nil
	}
	if len(a.shape) <= 1 {
		return a.Flatten().mustReshape(a.shape...)
	}
	newShape := append([]int(nil), a.shape...)
	for i, j := 0, len(newShape)-1; i < j; i, j = i+1, j-1 {
		newShape[i], newShape[j] = newShape[j], newShape[i]
	}
	out := make([]T, len(a.data))
	newArr := &NDArray[T]{data: out, shape: newShape, strides: computeStrides(newShape)}
	origIdx := make([]int, len(a.shape))
	for flat := range out {
		newIdx := unflattenIndex(flat, newShape, newArr.strides)
		for i := range origIdx {
			origIdx[i] = newIdx[len(newIdx)-1-i]
		}
		out[flat] = a.At(origIdx...)
	}
	return newArr
}

// ToSlice returns a copy of flat data.
func (a *NDArray[T]) ToSlice() []T {
	if a == nil {
		return nil
	}
	out := make([]T, len(a.data))
	copy(out, a.data)
	return out
}

func (a *NDArray[T]) flatIndexOrPanic(idx ...int) int {
	if len(idx) != len(a.shape) {
		panic(fmt.Sprintf("gods/ndarray: expected %d indices, got %d", len(a.shape), len(idx)))
	}
	flat := 0
	for i, v := range idx {
		if v < 0 || v >= a.shape[i] {
			panic(fmt.Sprintf("gods/ndarray: index %d out of bounds for axis %d with size %d", v, i, a.shape[i]))
		}
		flat += v * a.strides[i]
	}
	return flat
}

func (a *NDArray[T]) mustReshape(shape ...int) *NDArray[T] {
	out, err := a.Reshape(shape...)
	if err != nil {
		panic(err)
	}
	return out
}

func product(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	total := 1
	for _, dim := range shape {
		if dim < 0 {
			return 0
		}
		total *= dim
	}
	return total
}

func computeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func normalizeSliceSpec(spec SliceSpec, axisSize int) (start, stop, step int, err error) {
	step = 1
	if spec.Step != nil {
		step = *spec.Step
	}
	if step == 0 {
		return 0, 0, 0, fmt.Errorf("step cannot be zero")
	}
	if step > 0 {
		start = 0
		stop = axisSize
		if spec.Start != nil {
			start = clampIndex(*spec.Start, axisSize, true)
		}
		if spec.Stop != nil {
			stop = clampIndex(*spec.Stop, axisSize, true)
		}
	} else {
		start = axisSize - 1
		stop = -1
		if spec.Start != nil {
			start = clampIndex(*spec.Start, axisSize, false)
		}
		if spec.Stop != nil {
			stop = clampIndex(*spec.Stop, axisSize, false)
		}
	}
	return start, stop, step, nil
}

func clampIndex(v, n int, forward bool) int {
	if v < 0 {
		v += n
	}
	if forward {
		if v < 0 {
			return 0
		}
		if v > n {
			return n
		}
		return v
	}
	if v < -1 {
		return -1
	}
	if v >= n {
		return n - 1
	}
	return v
}

func unflattenIndex(flat int, shape, strides []int) []int {
	idx := make([]int, len(shape))
	for i := range shape {
		if strides[i] == 0 {
			idx[i] = 0
			continue
		}
		idx[i] = flat / strides[i]
		flat %= strides[i]
	}
	return idx
}

func almostZero(v float64) bool {
	return math.Abs(v) < 1e-12
}
