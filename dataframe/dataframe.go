package dataframe

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/kalpit-sharma-dev/gods/internal/util"
	"github.com/kalpit-sharma-dev/gods/series"
)

// column is the untyped internal interface all Series implement.
type column interface {
	Len() int
	Name() string
	Dtype() util.Dtype
	HasNulls() bool
	NullCount() int
	at(i int) (any, bool)
	filter(mask []bool) column
	clone() column
	withName(name string) column
	toSeriesAny() any
}

type seriesAdapter[T any] struct {
	s *series.Series[T]
}

func (c *seriesAdapter[T]) Len() int             { return c.s.Len() }
func (c *seriesAdapter[T]) Name() string         { return c.s.Name() }
func (c *seriesAdapter[T]) Dtype() util.Dtype    { return c.s.Dtype() }
func (c *seriesAdapter[T]) HasNulls() bool       { return c.s.HasNulls() }
func (c *seriesAdapter[T]) NullCount() int       { return c.s.NullCount() }
func (c *seriesAdapter[T]) at(i int) (any, bool) { return c.s.AtAny(i) }
func (c *seriesAdapter[T]) toSeriesAny() any     { return c.s }
func (c *seriesAdapter[T]) filter(mask []bool) column {
	v := c.s.FilterMask(mask)
	ns, _ := v.(*series.Series[T])
	return &seriesAdapter[T]{s: ns}
}
func (c *seriesAdapter[T]) clone() column {
	v := c.s.CloneAny()
	ns, _ := v.(*series.Series[T])
	return &seriesAdapter[T]{s: ns}
}
func (c *seriesAdapter[T]) withName(name string) column {
	v := c.s.WithName(name)
	ns, _ := v.(*series.Series[T])
	return &seriesAdapter[T]{s: ns}
}

// DataFrame is a columnar table of named typed columns.
type DataFrame struct {
	cols   []column
	index  map[string]int
	rowLen int
}

type compositeKey []any

func encodeCompositeKey(k compositeKey) string {
	parts := make([]string, 0, len(k))
	for _, v := range k {
		parts = append(parts, encodeKeyPart(v))
	}
	return strings.Join(parts, "\x1f")
}

func encodeKeyPart(v any) string {
	switch x := v.(type) {
	case nil:
		return "n:"
	case int:
		return "i:" + strconv.FormatInt(int64(x), 10)
	case int8:
		return "i:" + strconv.FormatInt(int64(x), 10)
	case int16:
		return "i:" + strconv.FormatInt(int64(x), 10)
	case int32:
		return "i:" + strconv.FormatInt(int64(x), 10)
	case int64:
		return "i:" + strconv.FormatInt(x, 10)
	case uint:
		return "u:" + strconv.FormatUint(uint64(x), 10)
	case uint8:
		return "u:" + strconv.FormatUint(uint64(x), 10)
	case uint16:
		return "u:" + strconv.FormatUint(uint64(x), 10)
	case uint32:
		return "u:" + strconv.FormatUint(uint64(x), 10)
	case uint64:
		return "u:" + strconv.FormatUint(x, 10)
	case float32:
		return "f:" + strconv.FormatUint(uint64(math.Float64bits(float64(x))), 16)
	case float64:
		return "f:" + strconv.FormatUint(math.Float64bits(x), 16)
	case bool:
		if x {
			return "b:1"
		}
		return "b:0"
	case string:
		return "s:" + strconv.Quote(x)
	case time.Time:
		return "t:" + x.UTC().Format(time.RFC3339Nano)
	default:
		return "x:" + fmt.Sprintf("%T=%v", v, v)
	}
}

func compareCompositeKey(a, b compositeKey) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		cmp := compareValues(a[i], b[i])
		if cmp != 0 {
			return cmp
		}
	}
	switch {
	case len(a) < len(b):
		return -1
	case len(a) > len(b):
		return 1
	default:
		return 0
	}
}

func compareInts(a int64, b any) int {
	switch x := b.(type) {
	case int:
		return compareInt64(a, int64(x))
	case int8:
		return compareInt64(a, int64(x))
	case int16:
		return compareInt64(a, int64(x))
	case int32:
		return compareInt64(a, int64(x))
	case int64:
		return compareInt64(a, x)
	case uint:
		if x > math.MaxInt64 {
			return -1
		}
		return compareInt64(a, int64(x))
	case uint8:
		return compareInt64(a, int64(x))
	case uint16:
		return compareInt64(a, int64(x))
	case uint32:
		return compareInt64(a, int64(x))
	case uint64:
		if x > math.MaxInt64 {
			return -1
		}
		return compareInt64(a, int64(x))
	default:
		return compareFloats(float64(a), toF64(b))
	}
}

func compareUints(a uint64, b any) int {
	switch x := b.(type) {
	case uint:
		return compareUint64(a, uint64(x))
	case uint8:
		return compareUint64(a, uint64(x))
	case uint16:
		return compareUint64(a, uint64(x))
	case uint32:
		return compareUint64(a, uint64(x))
	case uint64:
		return compareUint64(a, x)
	case int:
		if x < 0 {
			return 1
		}
		return compareUint64(a, uint64(x))
	case int8:
		if x < 0 {
			return 1
		}
		return compareUint64(a, uint64(x))
	case int16:
		if x < 0 {
			return 1
		}
		return compareUint64(a, uint64(x))
	case int32:
		if x < 0 {
			return 1
		}
		return compareUint64(a, uint64(x))
	case int64:
		if x < 0 {
			return 1
		}
		return compareUint64(a, uint64(x))
	default:
		return compareFloats(float64(a), toF64(b))
	}
}

func compareInt64(a, b int64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

func compareUint64(a, b uint64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

// New constructs a DataFrame from internal columns.
func New(columns ...column) (*DataFrame, error) {
	df := &DataFrame{
		cols:  make([]column, 0, len(columns)),
		index: make(map[string]int, len(columns)),
	}
	for i, c := range columns {
		if c == nil {
			return nil, fmt.Errorf("gods/dataframe: column %d is nil", i)
		}
		name := c.Name()
		if name == "" {
			return nil, fmt.Errorf("gods/dataframe: column %d has empty name", i)
		}
		if _, ok := df.index[name]; ok {
			return nil, fmt.Errorf("gods/dataframe: duplicate column name %q", name)
		}
		if i == 0 {
			df.rowLen = c.Len()
		} else if c.Len() != df.rowLen {
			return nil, fmt.Errorf("gods/dataframe: column %q length %d does not match row length %d", name, c.Len(), df.rowLen)
		}
		df.index[name] = len(df.cols)
		df.cols = append(df.cols, c.clone())
	}
	return df, nil
}

// FromMap constructs a DataFrame from map values of []T or *series.Series[T].
func FromMap(m map[string]any) (*DataFrame, error) {
	if len(m) == 0 {
		return Empty(), nil
	}
	names := make([]string, 0, len(m))
	for k := range m {
		names = append(names, k)
	}
	sort.Strings(names)
	cols := make([]column, 0, len(names))
	for _, name := range names {
		c, err := adaptToColumn(name, m[name])
		if err != nil {
			return nil, fmt.Errorf("gods/dataframe: column %q: %w", name, err)
		}
		cols = append(cols, c)
	}
	return New(cols...)
}

// Empty returns an empty DataFrame.
func Empty() *DataFrame {
	return &DataFrame{
		cols:   nil,
		index:  map[string]int{},
		rowLen: 0,
	}
}

// Shape returns DataFrame dimensions as (rows, cols).
func (df *DataFrame) Shape() (rows, cols int) {
	if df == nil {
		return 0, 0
	}
	return df.rowLen, len(df.cols)
}

// Columns returns ordered column names.
func (df *DataFrame) Columns() []string {
	if df == nil {
		return nil
	}
	out := make([]string, 0, len(df.cols))
	for _, c := range df.cols {
		out = append(out, c.Name())
	}
	return out
}

// Dtypes returns a map of column name to dtype.
func (df *DataFrame) Dtypes() map[string]util.Dtype {
	out := map[string]util.Dtype{}
	if df == nil {
		return out
	}
	for _, c := range df.cols {
		out[c.Name()] = c.Dtype()
	}
	return out
}

// String renders a compact tabular preview.
func (df *DataFrame) String() string {
	if df == nil {
		return "DataFrame<nil>"
	}
	var b strings.Builder
	rows, cols := df.Shape()
	fmt.Fprintf(&b, "DataFrame[%d x %d]\n", rows, cols)
	names := df.Columns()
	b.WriteString(strings.Join(names, "\t"))
	b.WriteString("\n")
	limit := rows
	if limit > 10 {
		limit = 10
	}
	for i := 0; i < limit; i++ {
		for j, c := range df.cols {
			v, ok := c.at(i)
			if !ok {
				b.WriteString("<null>")
			} else {
				fmt.Fprint(&b, v)
			}
			if j < len(df.cols)-1 {
				b.WriteString("\t")
			}
		}
		b.WriteString("\n")
	}
	if rows > limit {
		b.WriteString("...\n")
	}
	return b.String()
}

// Info prints shape, dtype, and null-count information.
func (df *DataFrame) Info() {
	if df == nil {
		fmt.Println("DataFrame<nil>")
		return
	}
	rows, cols := df.Shape()
	fmt.Printf("DataFrame: rows=%d cols=%d\n", rows, cols)
	for _, c := range df.cols {
		fmt.Printf("- %s: %s (nulls=%d)\n", c.Name(), c.Dtype(), c.NullCount())
	}
}

// Col returns a column by name.
func (df *DataFrame) Col(name string) (column, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	idx, ok := df.index[name]
	if !ok {
		return nil, fmt.Errorf("gods/dataframe: column %q not found (available: %v)", name, df.Columns())
	}
	return df.cols[idx], nil
}

// MustCol returns a column by name or panics.
func (df *DataFrame) MustCol(name string) column {
	c, err := df.Col(name)
	if err != nil {
		panic(err)
	}
	return c
}

// Select returns a DataFrame containing only the requested columns.
func (df *DataFrame) Select(names ...string) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	cols := make([]column, 0, len(names))
	for _, name := range names {
		c, err := df.Col(name)
		if err != nil {
			return nil, err
		}
		cols = append(cols, c.clone())
	}
	return New(cols...)
}

// Drop returns a DataFrame without the specified columns.
func (df *DataFrame) Drop(names ...string) *DataFrame {
	if df == nil {
		return nil
	}
	dropSet := map[string]struct{}{}
	for _, n := range names {
		dropSet[n] = struct{}{}
	}
	cols := make([]column, 0, len(df.cols))
	for _, c := range df.cols {
		if _, ok := dropSet[c.Name()]; ok {
			continue
		}
		cols = append(cols, c.clone())
	}
	out, err := New(cols...)
	if err != nil {
		return Empty()
	}
	return out
}

// AddColumn returns a new DataFrame with an appended column.
func (df *DataFrame) AddColumn(col column) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if col == nil {
		return nil, fmt.Errorf("gods/dataframe: nil column")
	}
	if _, ok := df.index[col.Name()]; ok {
		return nil, fmt.Errorf("gods/dataframe: column %q already exists", col.Name())
	}
	if len(df.cols) > 0 && col.Len() != df.rowLen {
		return nil, fmt.Errorf("gods/dataframe: new column length %d does not match row length %d", col.Len(), df.rowLen)
	}
	cols := make([]column, 0, len(df.cols)+1)
	for _, c := range df.cols {
		cols = append(cols, c.clone())
	}
	cols = append(cols, col.clone())
	return New(cols...)
}

// RenameColumn renames a column and returns a new DataFrame.
func (df *DataFrame) RenameColumn(old, new string) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if old == new {
		return df.Select(df.Columns()...)
	}
	if _, ok := df.index[new]; ok {
		return nil, fmt.Errorf("gods/dataframe: column %q already exists", new)
	}
	idx, ok := df.index[old]
	if !ok {
		return nil, fmt.Errorf("gods/dataframe: column %q not found", old)
	}
	cols := make([]column, len(df.cols))
	for i, c := range df.cols {
		if i == idx {
			cols[i] = c.withName(new)
		} else {
			cols[i] = c.clone()
		}
	}
	return New(cols...)
}

// Row returns row i as a map.
func (df *DataFrame) Row(i int) (map[string]any, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if i < 0 || i >= df.rowLen {
		return nil, fmt.Errorf("gods/dataframe: row index %d out of bounds [0,%d)", i, df.rowLen)
	}
	row := make(map[string]any, len(df.cols))
	for _, c := range df.cols {
		v, ok := c.at(i)
		if !ok {
			row[c.Name()] = nil
		} else {
			row[c.Name()] = v
		}
	}
	return row, nil
}

// Head returns the first n rows.
func (df *DataFrame) Head(n int) *DataFrame {
	if df == nil {
		return nil
	}
	if n < 0 {
		n = 0
	}
	if n > df.rowLen {
		n = df.rowLen
	}
	mask := make([]bool, df.rowLen)
	for i := 0; i < n; i++ {
		mask[i] = true
	}
	out := df.filterWithMask(mask)
	if out == nil {
		return Empty()
	}
	return out
}

// Tail returns the last n rows.
func (df *DataFrame) Tail(n int) *DataFrame {
	if df == nil {
		return nil
	}
	if n < 0 {
		n = 0
	}
	if n > df.rowLen {
		n = df.rowLen
	}
	mask := make([]bool, df.rowLen)
	for i := df.rowLen - n; i < df.rowLen; i++ {
		if i >= 0 {
			mask[i] = true
		}
	}
	out := df.filterWithMask(mask)
	if out == nil {
		return Empty()
	}
	return out
}

// Sample returns n random rows (without replacement) using seed.
func (df *DataFrame) Sample(n int, seed int64) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	if n < 0 {
		return nil, fmt.Errorf("gods/dataframe: sample size must be >= 0")
	}
	if n > df.rowLen {
		return nil, fmt.Errorf("gods/dataframe: sample size %d exceeds row count %d", n, df.rowLen)
	}
	if n == 0 {
		return df.Head(0), nil
	}
	r := rand.New(rand.NewSource(seed))
	perm := r.Perm(df.rowLen)[:n]
	sort.Ints(perm)
	return df.selectRows(perm)
}

func (df *DataFrame) filterWithMask(mask []bool) *DataFrame {
	cols := make([]column, len(df.cols))
	for i, c := range df.cols {
		cols[i] = c.filter(mask)
	}
	out, err := New(cols...)
	if err != nil {
		return Empty()
	}
	return out
}

func (df *DataFrame) selectRows(indices []int) (*DataFrame, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	for _, idx := range indices {
		if idx < 0 || idx >= df.rowLen {
			return nil, fmt.Errorf("gods/dataframe: row index %d out of bounds [0,%d)", idx, df.rowLen)
		}
	}
	cols := make([]column, len(df.cols))
	for i, c := range df.cols {
		next, err := takeColumnRows(c, indices)
		if err != nil {
			return nil, fmt.Errorf("gods/dataframe: take rows for column %q: %w", c.Name(), err)
		}
		cols[i] = next
	}
	return New(cols...)
}

func dataFrameFromRows(rows []map[string]any) (*DataFrame, error) {
	if len(rows) == 0 {
		return Empty(), nil
	}
	colSet := map[string]struct{}{}
	for _, row := range rows {
		for k := range row {
			colSet[k] = struct{}{}
		}
	}
	cols := make([]string, 0, len(colSet))
	for c := range colSet {
		cols = append(cols, c)
	}
	sort.Strings(cols)
	builtCols := make([]column, 0, len(cols))
	for _, name := range cols {
		values := make([]any, len(rows))
		nullMask := make([]bool, len(rows))
		for i, row := range rows {
			v, ok := row[name]
			if !ok || v == nil {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		col, err := inferSeriesColumn(name, values, nullMask)
		if err != nil {
			return nil, fmt.Errorf("gods/dataframe: infer column %q from rows: %w", name, err)
		}
		builtCols = append(builtCols, col)
	}
	return New(builtCols...)
}

func adaptToColumn(name string, v any) (column, error) {
	switch x := v.(type) {
	case *series.Series[int64]:
		return &seriesAdapter[int64]{s: toNamedSeries(name, x)}, nil
	case *series.Series[float64]:
		return &seriesAdapter[float64]{s: toNamedSeries(name, x)}, nil
	case *series.Series[bool]:
		return &seriesAdapter[bool]{s: toNamedSeries(name, x)}, nil
	case *series.Series[string]:
		return &seriesAdapter[string]{s: toNamedSeries(name, x)}, nil
	case *series.Series[time.Time]:
		return &seriesAdapter[time.Time]{s: toNamedSeries(name, x)}, nil
	case *series.Series[any]:
		return &seriesAdapter[any]{s: toNamedSeries(name, x)}, nil
	case []int64:
		return &seriesAdapter[int64]{s: series.New(name, x)}, nil
	case []float64:
		return &seriesAdapter[float64]{s: series.New(name, x)}, nil
	case []bool:
		return &seriesAdapter[bool]{s: series.New(name, x)}, nil
	case []string:
		return &seriesAdapter[string]{s: series.New(name, x)}, nil
	case []time.Time:
		return &seriesAdapter[time.Time]{s: series.New(name, x)}, nil
	case []any:
		return &seriesAdapter[any]{s: series.New(name, x)}, nil
	default:
		return nil, fmt.Errorf("unsupported column value type %T", v)
	}
}

func toNamedSeries[T any](name string, s *series.Series[T]) *series.Series[T] {
	if s == nil {
		return series.New[T](name, nil)
	}
	if s.Name() == name {
		return series.MustAs[T](s.CloneAny())
	}
	return series.MustAs[T](s.WithName(name))
}

func inferSeriesColumn(name string, values []any, nullMask []bool) (column, error) {
	first := -1
	for i, v := range values {
		if i < len(nullMask) && nullMask[i] {
			continue
		}
		if v == nil {
			continue
		}
		first = i
		break
	}
	if first == -1 {
		s := series.WithNulls[any](name, values, nullMask)
		return &seriesAdapter[any]{s: s}, nil
	}
	switch values[first].(type) {
	case int:
		vals := make([]int64, len(values))
		mask := make([]bool, len(values))
		for i, v := range values {
			if i < len(nullMask) && nullMask[i] || v == nil {
				mask[i] = true
				continue
			}
			n, ok := numericToInt64(v)
			if !ok {
				return nil, fmt.Errorf("gods/dataframe: cannot coerce value %T to int64 for column %q", v, name)
			}
			vals[i] = n
		}
		return &seriesAdapter[int64]{s: series.WithNulls(name, vals, mask)}, nil
	case int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		vals := make([]int64, len(values))
		mask := make([]bool, len(values))
		for i, v := range values {
			if i < len(nullMask) && nullMask[i] || v == nil {
				mask[i] = true
				continue
			}
			n, ok := numericToInt64(v)
			if !ok {
				return nil, fmt.Errorf("gods/dataframe: cannot coerce value %T to int64 for column %q", v, name)
			}
			vals[i] = n
		}
		return &seriesAdapter[int64]{s: series.WithNulls(name, vals, mask)}, nil
	case float32, float64:
		vals := make([]float64, len(values))
		mask := make([]bool, len(values))
		for i, v := range values {
			if i < len(nullMask) && nullMask[i] || v == nil {
				mask[i] = true
				continue
			}
			n, ok := numericToFloat64(v)
			if !ok {
				return nil, fmt.Errorf("gods/dataframe: cannot coerce value %T to float64 for column %q", v, name)
			}
			vals[i] = n
		}
		return &seriesAdapter[float64]{s: series.WithNulls(name, vals, mask)}, nil
	case bool:
		vals := make([]bool, len(values))
		mask := make([]bool, len(values))
		for i, v := range values {
			if i < len(nullMask) && nullMask[i] || v == nil {
				mask[i] = true
				continue
			}
			b, ok := v.(bool)
			if !ok {
				return nil, fmt.Errorf("gods/dataframe: cannot coerce value %T to bool for column %q", v, name)
			}
			vals[i] = b
		}
		return &seriesAdapter[bool]{s: series.WithNulls(name, vals, mask)}, nil
	case string:
		vals := make([]string, len(values))
		mask := make([]bool, len(values))
		for i, v := range values {
			if i < len(nullMask) && nullMask[i] || v == nil {
				mask[i] = true
				continue
			}
			sv, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("gods/dataframe: cannot coerce value %T to string for column %q", v, name)
			}
			vals[i] = sv
		}
		return &seriesAdapter[string]{s: series.WithNulls(name, vals, mask)}, nil
	case time.Time:
		vals := make([]time.Time, len(values))
		mask := make([]bool, len(values))
		for i, v := range values {
			if i < len(nullMask) && nullMask[i] || v == nil {
				mask[i] = true
				continue
			}
			tv, ok := v.(time.Time)
			if !ok {
				return nil, fmt.Errorf("gods/dataframe: cannot coerce value %T to time.Time for column %q", v, name)
			}
			vals[i] = tv
		}
		return &seriesAdapter[time.Time]{s: series.WithNulls(name, vals, mask)}, nil
	default:
		s := series.WithNulls[any](name, values, nullMask)
		return &seriesAdapter[any]{s: s}, nil
	}
}

func numericToInt64(v any) (int64, bool) {
	switch x := v.(type) {
	case int:
		return int64(x), true
	case int8:
		return int64(x), true
	case int16:
		return int64(x), true
	case int32:
		return int64(x), true
	case int64:
		return x, true
	case uint:
		return int64(x), true
	case uint8:
		return int64(x), true
	case uint16:
		return int64(x), true
	case uint32:
		return int64(x), true
	case uint64:
		return int64(x), true
	case float32:
		return int64(x), true
	case float64:
		return int64(x), true
	default:
		return 0, false
	}
}

func numericToFloat64(v any) (float64, bool) {
	switch x := v.(type) {
	case int:
		return float64(x), true
	case int8:
		return float64(x), true
	case int16:
		return float64(x), true
	case int32:
		return float64(x), true
	case int64:
		return float64(x), true
	case uint:
		return float64(x), true
	case uint8:
		return float64(x), true
	case uint16:
		return float64(x), true
	case uint32:
		return float64(x), true
	case uint64:
		return float64(x), true
	case float32:
		return float64(x), true
	case float64:
		return x, true
	default:
		return 0, false
	}
}

func takeColumnRows(c column, indices []int) (column, error) {
	switch s := c.toSeriesAny().(type) {
	case *series.Series[int64]:
		values := make([]int64, len(indices))
		nullMask := make([]bool, len(indices))
		for i, idx := range indices {
			v, ok := s.At(idx)
			if !ok {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		return &seriesAdapter[int64]{s: series.WithNulls(s.Name(), values, nullMask)}, nil
	case *series.Series[float64]:
		values := make([]float64, len(indices))
		nullMask := make([]bool, len(indices))
		for i, idx := range indices {
			v, ok := s.At(idx)
			if !ok {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		return &seriesAdapter[float64]{s: series.WithNulls(s.Name(), values, nullMask)}, nil
	case *series.Series[bool]:
		values := make([]bool, len(indices))
		nullMask := make([]bool, len(indices))
		for i, idx := range indices {
			v, ok := s.At(idx)
			if !ok {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		return &seriesAdapter[bool]{s: series.WithNulls(s.Name(), values, nullMask)}, nil
	case *series.Series[string]:
		values := make([]string, len(indices))
		nullMask := make([]bool, len(indices))
		for i, idx := range indices {
			v, ok := s.At(idx)
			if !ok {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		return &seriesAdapter[string]{s: series.WithNulls(s.Name(), values, nullMask)}, nil
	case *series.Series[time.Time]:
		values := make([]time.Time, len(indices))
		nullMask := make([]bool, len(indices))
		for i, idx := range indices {
			v, ok := s.At(idx)
			if !ok {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		return &seriesAdapter[time.Time]{s: series.WithNulls(s.Name(), values, nullMask)}, nil
	case *series.Series[any]:
		values := make([]any, len(indices))
		nullMask := make([]bool, len(indices))
		for i, idx := range indices {
			v, ok := s.At(idx)
			if !ok {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		return &seriesAdapter[any]{s: series.WithNulls(s.Name(), values, nullMask)}, nil
	default:
		return nil, fmt.Errorf("unsupported column series type %T", c.toSeriesAny())
	}
}

// SeriesAt returns a typed series pointer for the provided column name.
func SeriesAt[T any](df *DataFrame, name string) (*series.Series[T], error) {
	if df == nil {
		return nil, fmt.Errorf("gods/dataframe: nil dataframe")
	}
	c, err := df.Col(name)
	if err != nil {
		return nil, err
	}
	switch x := c.toSeriesAny().(type) {
	case *series.Series[T]:
		return x, nil
	default:
		return nil, fmt.Errorf("gods/dataframe: column %q is %T, not requested series type", name, c.toSeriesAny())
	}
}

// MustSeriesAt returns a typed series pointer or panics.
func MustSeriesAt[T any](df *DataFrame, name string) *series.Series[T] {
	s, err := SeriesAt[T](df, name)
	if err != nil {
		panic(err)
	}
	return s
}
