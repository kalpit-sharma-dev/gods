package series

import (
	"fmt"
	"strings"
	"time"

	"github.com/kalpit-sharma-dev/gods/internal/bitmap"
	"github.com/kalpit-sharma-dev/gods/internal/util"
)

// Series is a typed 1D column with a validity bitmap.
type Series[T any] struct {
	name     string
	values   []T
	validity *bitmap.Bitmap
	dtype    util.Dtype
}

// New creates a named series with all values marked valid.
func New[T any](name string, values []T) *Series[T] {
	copied := make([]T, len(values))
	copy(copied, values)
	return &Series[T]{
		name:     name,
		values:   copied,
		validity: bitmap.New(len(values)),
		dtype:    inferDtype[T](),
	}
}

// WithNulls creates a named series using a null mask where true means null.
func WithNulls[T any](name string, values []T, nullMask []bool) *Series[T] {
	s := New(name, values)
	for i := 0; i < len(s.values) && i < len(nullMask); i++ {
		if nullMask[i] {
			s.validity.Set(i, false)
		}
	}
	return s
}

// FromSlice creates an unnamed series from a value slice.
func FromSlice[T any](values []T) *Series[T] {
	return New("", values)
}

// Name returns the series name.
func (s *Series[T]) Name() string {
	if s == nil {
		return ""
	}
	return s.name
}

// Len returns the number of rows.
func (s *Series[T]) Len() int {
	if s == nil {
		return 0
	}
	return len(s.values)
}

// Dtype returns the series dtype.
func (s *Series[T]) Dtype() util.Dtype {
	if s == nil {
		return util.DtypeAny
	}
	return s.dtype
}

// HasNulls reports whether this series contains null values.
func (s *Series[T]) HasNulls() bool {
	if s == nil {
		return false
	}
	return s.validity.NullCount() > 0
}

// NullCount returns number of null elements.
func (s *Series[T]) NullCount() int {
	if s == nil {
		return 0
	}
	return s.validity.NullCount()
}

// String renders a compact preview of the series.
func (s *Series[T]) String() string {
	if s == nil {
		return "Series<nil>"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Series[%s](name=%q, len=%d)\n", s.dtype, s.name, s.Len())
	limit := s.Len()
	if limit > 10 {
		limit = 10
	}
	for i := 0; i < limit; i++ {
		v, ok := s.At(i)
		if !ok {
			fmt.Fprintf(&b, "[%d] <null>\n", i)
			continue
		}
		fmt.Fprintf(&b, "[%d] %v\n", i, v)
	}
	if s.Len() > limit {
		b.WriteString("...\n")
	}
	return b.String()
}

// At returns value and validity at index i.
func (s *Series[T]) At(i int) (T, bool) {
	var zero T
	if s == nil || i < 0 || i >= len(s.values) {
		return zero, false
	}
	return s.values[i], s.validity.IsValid(i)
}

// MustAt returns value at index i and panics if null or out of bounds.
func (s *Series[T]) MustAt(i int) T {
	v, ok := s.At(i)
	if !ok {
		panic(fmt.Sprintf("gods/series: index %d is null or out of bounds", i))
	}
	return v
}

// Set sets value at index i and marks it valid.
func (s *Series[T]) Set(i int, val T) {
	if s == nil || i < 0 || i >= len(s.values) {
		return
	}
	s.values[i] = val
	s.validity.Set(i, true)
}

// SetNull marks the index i as null.
func (s *Series[T]) SetNull(i int) {
	if s == nil {
		return
	}
	s.validity.Set(i, false)
}

// Values returns a copy of values.
func (s *Series[T]) Values() []T {
	if s == nil {
		return nil
	}
	out := make([]T, len(s.values))
	copy(out, s.values)
	return out
}

// ValidValues returns only valid values in order.
func (s *Series[T]) ValidValues() []T {
	if s == nil {
		return nil
	}
	out := make([]T, 0, s.Len()-s.NullCount())
	for i := range s.values {
		if s.validity.IsValid(i) {
			out = append(out, s.values[i])
		}
	}
	return out
}

// ToSlice returns a copy of values.
func (s *Series[T]) ToSlice() []T {
	return s.Values()
}

// AtAny returns value and validity at index i as an untyped value.
func (s *Series[T]) AtAny(i int) (any, bool) {
	v, ok := s.At(i)
	return any(v), ok
}

// FilterMask returns a new series filtered by a row mask.
// A true mask entry keeps the corresponding row.
func (s *Series[T]) FilterMask(mask []bool) any {
	if s == nil {
		return (*Series[T])(nil)
	}
	values := make([]T, 0, len(mask))
	nullMask := make([]bool, 0, len(mask))
	n := len(mask)
	if n > s.Len() {
		n = s.Len()
	}
	for i := 0; i < n; i++ {
		if !mask[i] {
			continue
		}
		values = append(values, s.values[i])
		nullMask = append(nullMask, !s.validity.IsValid(i))
	}
	return WithNulls(s.name, values, nullMask)
}

// CloneAny returns a deep clone as an untyped value.
func (s *Series[T]) CloneAny() any {
	if s == nil {
		return (*Series[T])(nil)
	}
	return s.cloneWith(s.values, s.validity)
}

// WithName returns a cloned series with a new name.
func (s *Series[T]) WithName(name string) any {
	if s == nil {
		return (*Series[T])(nil)
	}
	out := s.cloneWith(s.values, s.validity)
	out.name = name
	return out
}

func (s *Series[T]) cloneWith(values []T, valid *bitmap.Bitmap) *Series[T] {
	copied := make([]T, len(values))
	copy(copied, values)
	return &Series[T]{
		name:     s.name,
		values:   copied,
		validity: valid.Clone(),
		dtype:    s.dtype,
	}
}

func inferDtype[T any]() util.Dtype {
	var z T
	switch any(z).(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return util.DtypeInt64
	case float32, float64:
		return util.DtypeFloat64
	case bool:
		return util.DtypeBool
	case string:
		return util.DtypeString
	case time.Time:
		return util.DtypeTime
	default:
		return util.DtypeAny
	}
}

// MustAs converts a dynamic value into a typed series or panics.
func MustAs[T any](v any) *Series[T] {
	s, ok := v.(*Series[T])
	if !ok {
		if unwrappable, canUnwrap := v.(interface{ ToSeriesAny() any }); canUnwrap {
			if inner, okInner := unwrappable.ToSeriesAny().(*Series[T]); okInner {
				return inner
			}
		}
		panic(fmt.Sprintf("gods/series: value is %T, not *Series[%T]", v, *new(T)))
	}
	return s
}
