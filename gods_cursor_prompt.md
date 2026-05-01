# GoDS — Go Data Science Library
## Complete End-to-End Cursor Prompt

> Paste this entire document into Cursor's Composer (Cmd+I) or use it as a `.cursorrules` project file.

---

## ROLE & MISSION

You are a senior Go systems engineer and numerical computing expert building `gods` — a production-grade data manipulation and numerical computing library for Go, inspired by pandas and NumPy but idiomatic to Go.

This is NOT a Python port. Design every API around Go's strengths: strong typing, interfaces, generics (Go 1.21+), goroutines, and zero-overhead abstractions.

**Go version**: 1.21+
**Module name**: `github.com/yourusername/gods`
**No external dependencies** unless absolutely necessary (gonum for BLAS is acceptable).

---

## PHASE 1 — PROJECT SCAFFOLD

Create the following directory structure. Do not deviate from it.

```
gods/
├── go.mod
├── go.sum
├── README.md
├── series/
│   ├── series.go
│   ├── series_ops.go
│   ├── series_stats.go
│   ├── series_null.go
│   └── series_test.go
├── dataframe/
│   ├── dataframe.go
│   ├── dataframe_filter.go
│   ├── dataframe_join.go
│   ├── dataframe_groupby.go
│   ├── dataframe_sort.go
│   └── dataframe_test.go
├── ndarray/
│   ├── ndarray.go
│   ├── ndarray_ops.go
│   ├── ndarray_linalg.go
│   └── ndarray_test.go
├── io/
│   ├── csv.go
│   ├── json.go
│   ├── parquet_iface.go   ← interface only, no impl
│   └── io_test.go
├── stats/
│   ├── descriptive.go
│   ├── distributions.go
│   └── stats_test.go
├── parallel/
│   ├── pool.go
│   ├── parallel_ops.go
│   └── parallel_test.go
├── expr/
│   ├── expr.go            ← query expression DSL
│   └── expr_test.go
└── internal/
    ├── bitmap/
    │   └── bitmap.go      ← null bitmask
    └── util/
        └── util.go
```

`go.mod`:
```
module github.com/yourusername/gods

go 1.21

require (
    golang.org/x/exp v0.0.0-20240119083558-1b970713d09a
)
```

---

## PHASE 2 — CORE TYPE SYSTEM

### 2.1 Null/Validity System (`internal/bitmap/bitmap.go`)

Use a **bitmask validity array** — NOT pointer-based nullables. This is the approach used by Apache Arrow and is cache-friendly.

```go
package bitmap

// Bitmap is a compact validity mask. Bit i = 1 means valid, 0 means null.
type Bitmap struct {
    data []uint64
    len  int
}

func New(n int) *Bitmap
func (b *Bitmap) Set(i int, valid bool)
func (b *Bitmap) IsValid(i int) bool
func (b *Bitmap) IsNull(i int) bool
func (b *Bitmap) ValidCount() int
func (b *Bitmap) NullCount() int
func (b *Bitmap) Clone() *Bitmap
```

Rules:
- Bit operations only — no `bool` slices
- `data` length = `ceil(len / 64)`
- Thread-safe reads are fine; writes need external sync

---

### 2.2 Dtype System (`internal/util/util.go`)

```go
package util

type Dtype int

const (
    DtypeInt64 Dtype = iota
    DtypeFloat64
    DtypeBool
    DtypeString
    DtypeTime
    DtypeAny
)

func (d Dtype) String() string
func (d Dtype) IsNumeric() bool
```

---

## PHASE 3 — SERIES (`series/series.go`)

Series is a typed, nullable, 1D array. It is the atomic unit of all data structures.

### 3.1 Core Type

```go
package series

import (
    "github.com/yourusername/gods/internal/bitmap"
    "github.com/yourusername/gods/internal/util"
)

// Series is a typed 1D column with a validity bitmap.
// T must be int64 | float64 | bool | string | time.Time | any
type Series[T any] struct {
    name     string
    values   []T
    validity *bitmap.Bitmap
    dtype    util.Dtype
}

// Constructors
func New[T any](name string, values []T) *Series[T]
func WithNulls[T any](name string, values []T, nullMask []bool) *Series[T]
func FromSlice[T any](values []T) *Series[T]

// Metadata
func (s *Series[T]) Name() string
func (s *Series[T]) Len() int
func (s *Series[T]) Dtype() util.Dtype
func (s *Series[T]) HasNulls() bool
func (s *Series[T]) NullCount() int
func (s *Series[T]) String() string   // pretty-print first 10 rows
```

### 3.2 Access & Mutation (`series/series.go`)

```go
func (s *Series[T]) At(i int) (T, bool)        // (value, isValid)
func (s *Series[T]) MustAt(i int) T             // panics if null
func (s *Series[T]) Set(i int, val T)
func (s *Series[T]) SetNull(i int)
func (s *Series[T]) Values() []T                // returns a copy
func (s *Series[T]) ValidValues() []T           // only non-null values
func (s *Series[T]) ToSlice() []T               // alias for Values()
```

### 3.3 Functional Operations (`series/series_ops.go`)

```go
// Map applies fn to every non-null element; nulls propagate.
func (s *Series[T]) Map(fn func(T) T) *Series[T]

// MapErr is like Map but fn can return an error; errors become nulls.
func (s *Series[T]) MapErr(fn func(T) (T, error)) (*Series[T], []error)

// Filter returns a new Series with elements where fn returns true.
func (s *Series[T]) Filter(fn func(T) bool) *Series[T]

// Reduce folds the series. Skips nulls.
func (s *Series[T]) Reduce(initial T, fn func(acc, val T) T) T

// Apply mutates in-place. Returns self for chaining.
func (s *Series[T]) Apply(fn func(T) T) *Series[T]

// Head/Tail
func (s *Series[T]) Head(n int) *Series[T]
func (s *Series[T]) Tail(n int) *Series[T]

// Concat merges two same-typed series.
func Concat[T any](a, b *Series[T]) *Series[T]

// Cast attempts to convert to target Dtype. Returns error if incompatible.
func (s *Series[T]) Cast(dtype util.Dtype) (any, error)
```

### 3.4 Null Handling (`series/series_null.go`)

```go
// IsNull returns a bool Series indicating null positions.
func (s *Series[T]) IsNull() *Series[bool]

// IsValid is the inverse of IsNull.
func (s *Series[T]) IsValid() *Series[bool]

// FillNa replaces nulls with fillValue.
func (s *Series[T]) FillNa(fillValue T) *Series[T]

// FillNaForward fills nulls with the last valid value (forward fill).
func (s *Series[T]) FillNaForward() *Series[T]

// FillNaBackward fills nulls with the next valid value (backward fill).
func (s *Series[T]) FillNaBackward() *Series[T]

// DropNa returns a new Series with null rows removed.
func (s *Series[T]) DropNa() *Series[T]

// Mask sets elements to null where condition is true.
func (s *Series[T]) Mask(condition *Series[bool]) *Series[T]
```

### 3.5 Statistical Operations (`series/series_stats.go`)

Only implement for numeric types. Use a type constraint helper internally.

```go
// All stats skip null values. Return (result, error) where error is non-nil
// if the series is empty or all-null.

func Sum[T constraints.Number](s *Series[T]) (T, error)
func Mean[T constraints.Number](s *Series[T]) (float64, error)
func Median[T constraints.Number](s *Series[T]) (float64, error)
func Std[T constraints.Number](s *Series[T]) (float64, error)    // sample std dev (n-1)
func Var[T constraints.Number](s *Series[T]) (float64, error)    // sample variance
func Min[T constraints.Number](s *Series[T]) (T, error)
func Max[T constraints.Number](s *Series[T]) (T, error)
func ArgMin[T constraints.Number](s *Series[T]) (int, error)     // index of min
func ArgMax[T constraints.Number](s *Series[T]) (int, error)     // index of max
func Quantile[T constraints.Number](s *Series[T], q float64) (float64, error) // q in [0,1]

// Describe returns a map of stat_name → value for quick summary.
func Describe[T constraints.Number](s *Series[T]) map[string]float64
```

---

## PHASE 4 — DATAFRAME (`dataframe/dataframe.go`)

DataFrame is a collection of named Series with a shared row index. Internally it is **columnar** — columns are stored as `[]any` backed by `*Series[T]` values, accessed via interface.

### 4.1 Internal Column Interface

```go
// column is the untyped internal interface all Series implement.
type column interface {
    Len() int
    Name() string
    Dtype() util.Dtype
    HasNulls() bool
    NullCount() int
    at(i int) (any, bool)        // lowercase — internal only
    filter(mask []bool) column
    clone() column
}
```

### 4.2 Core DataFrame Type

```go
package dataframe

type DataFrame struct {
    cols    []column          // ordered columns
    index   map[string]int    // column name → col index
    rowLen  int
}

// Constructors
func New(columns ...column) (*DataFrame, error)
func FromMap(m map[string]any) (*DataFrame, error)   // any = []T or *Series[T]
func Empty() *DataFrame

// Metadata
func (df *DataFrame) Shape() (rows, cols int)
func (df *DataFrame) Columns() []string
func (df *DataFrame) Dtypes() map[string]util.Dtype
func (df *DataFrame) String() string          // tabular pretty-print
func (df *DataFrame) Info()                   // prints to stdout: shape, dtypes, null counts
```

### 4.3 Column Selection & Access

```go
func (df *DataFrame) Col(name string) (column, error)
func (df *DataFrame) MustCol(name string) column          // panics if missing
func (df *DataFrame) Select(names ...string) (*DataFrame, error)
func (df *DataFrame) Drop(names ...string) *DataFrame
func (df *DataFrame) AddColumn(col column) (*DataFrame, error)
func (df *DataFrame) RenameColumn(old, new string) (*DataFrame, error)
func (df *DataFrame) Row(i int) (map[string]any, error)   // row as map
func (df *DataFrame) Head(n int) *DataFrame
func (df *DataFrame) Tail(n int) *DataFrame
func (df *DataFrame) Sample(n int, seed int64) (*DataFrame, error)
```

### 4.4 Filtering (`dataframe/dataframe_filter.go`)

```go
// Row is passed to the filter predicate for row-by-row queries.
type Row map[string]any

// Filter returns rows where predicate returns true.
func (df *DataFrame) Filter(predicate func(Row) bool) *DataFrame

// FilterExpr uses the expression DSL (see Phase 7).
func (df *DataFrame) FilterExpr(e expr.Expr) (*DataFrame, error)

// BoolMask applies a pre-computed boolean Series as a row mask.
func (df *DataFrame) BoolMask(mask *series.Series[bool]) (*DataFrame, error)

// DropNullRows drops rows that have any null in the specified columns.
// If no columns specified, checks all columns.
func (df *DataFrame) DropNullRows(columns ...string) *DataFrame

// FillNa fills nulls in a column with a scalar value.
func (df *DataFrame) FillNa(column string, value any) (*DataFrame, error)
```

### 4.5 Transformation

```go
// Apply applies fn to every element of column; returns a new DataFrame.
// fn receives (any, bool) = (value, isValid).
func (df *DataFrame) Apply(column string, fn func(any, bool) (any, bool)) (*DataFrame, error)

// Assign creates or replaces a column using a row-wise function.
func (df *DataFrame) Assign(newColName string, fn func(Row) (any, bool)) (*DataFrame, error)

// Mutate is like Assign but modifies in-place and returns self.
func (df *DataFrame) Mutate(name string, fn func(Row) (any, bool)) error

// Pipe chains DataFrame → DataFrame transformations.
func (df *DataFrame) Pipe(fns ...func(*DataFrame) (*DataFrame, error)) (*DataFrame, error)
```

### 4.6 Sorting (`dataframe/dataframe_sort.go`)

```go
type SortKey struct {
    Column    string
    Ascending bool
    NullsLast bool
}

// Sort returns a sorted DataFrame by one or more keys.
func (df *DataFrame) Sort(keys ...SortKey) (*DataFrame, error)

// Rank returns a new float64 Series with rank of each element in column.
// method: "average" | "min" | "max" | "dense"
func (df *DataFrame) Rank(column, method string) (*series.Series[float64], error)
```

### 4.7 GroupBy & Aggregation (`dataframe/dataframe_groupby.go`)

```go
type AggFunc string

const (
    AggSum   AggFunc = "sum"
    AggMean  AggFunc = "mean"
    AggMin   AggFunc = "min"
    AggMax   AggFunc = "max"
    AggCount AggFunc = "count"
    AggStd   AggFunc = "std"
    AggFirst AggFunc = "first"
    AggLast  AggFunc = "last"
)

type GroupBy struct {
    df      *DataFrame
    byKeys  []string
    groups  map[string][]int  // group key → row indices
}

func (df *DataFrame) GroupBy(columns ...string) (*GroupBy, error)

// Agg aggregates each column using the specified function.
// map key = column name, value = AggFunc to apply.
func (g *GroupBy) Agg(spec map[string]AggFunc) (*DataFrame, error)

// Apply applies a custom function to each group's sub-DataFrame.
func (g *GroupBy) Apply(fn func(group *DataFrame) (*DataFrame, error)) (*DataFrame, error)

// Size returns a Series with the count of rows per group.
func (g *GroupBy) Size() (*series.Series[int64], error)

// Transform applies a function to each group and returns a full-length result,
// aligned with the original DataFrame's row order.
func (g *GroupBy) Transform(column string, fn func(*DataFrame) any) (*series.Series[any], error)
```

### 4.8 Joins (`dataframe/dataframe_join.go`)

```go
type JoinType string

const (
    InnerJoin JoinType = "inner"
    LeftJoin  JoinType = "left"
    RightJoin JoinType = "right"
    OuterJoin JoinType = "outer"
)

// Join performs a join on one or more key columns.
// On: column names that exist in both DataFrames.
// Suffix: appended to duplicate non-key column names (default: ["_x", "_y"]).
func Join(left, right *DataFrame, on []string, how JoinType, suffixes [2]string) (*DataFrame, error)

// Concat stacks DataFrames vertically. Columns must match or be a superset.
func Concat(dfs []*DataFrame, ignoreIndex bool) (*DataFrame, error)
```

---

## PHASE 5 — NDARRAY (`ndarray/ndarray.go`)

NDArray is a typed N-dimensional array backed by a flat slice. All indexing is row-major (C order).

### 5.1 Core Type

```go
package ndarray

import "golang.org/x/exp/constraints"

type NDArray[T constraints.Number] struct {
    data    []T
    shape   []int
    strides []int   // pre-computed strides for indexing
}

// Constructors
func Zeros[T constraints.Number](shape ...int) *NDArray[T]
func Ones[T constraints.Number](shape ...int) *NDArray[T]
func Full[T constraints.Number](value T, shape ...int) *NDArray[T]
func FromSlice[T constraints.Number](data []T, shape ...int) (*NDArray[T], error)
func Arange[T constraints.Number](start, stop, step T) *NDArray[T]
func Linspace(start, stop float64, n int) *NDArray[float64]
func Random(shape ...int) *NDArray[float64]          // uniform [0, 1)
func RandomNormal(mean, std float64, shape ...int) *NDArray[float64]

// Metadata
func (a *NDArray[T]) Shape() []int
func (a *NDArray[T]) Ndim() int
func (a *NDArray[T]) Size() int          // total element count
func (a *NDArray[T]) Strides() []int
func (a *NDArray[T]) String() string     // pretty-print
```

### 5.2 Indexing

```go
// At returns element at multi-dimensional index. Panics on out-of-bounds.
func (a *NDArray[T]) At(idx ...int) T

// Set sets element at multi-dimensional index.
func (a *NDArray[T]) Set(val T, idx ...int)

// SliceSpec describes a single-axis slice: [Start:Stop:Step]. nil = full axis.
type SliceSpec struct{ Start, Stop, Step *int }

func (a *NDArray[T]) Slice(specs ...SliceSpec) (*NDArray[T], error)

// Flatten returns a 1D copy.
func (a *NDArray[T]) Flatten() *NDArray[T]

// Reshape returns a view with new shape. Total size must match.
func (a *NDArray[T]) Reshape(newShape ...int) (*NDArray[T], error)

// T returns the transpose (2D generalized to N-D via axis reversal).
func (a *NDArray[T]) T() *NDArray[T]

// ToSlice returns the underlying flat slice (copy).
func (a *NDArray[T]) ToSlice() []T
```

### 5.3 Element-wise Operations (`ndarray/ndarray_ops.go`)

All ops follow NumPy broadcasting semantics: shapes must be broadcastable.

```go
// Scalar ops
func (a *NDArray[T]) AddScalar(s T) *NDArray[T]
func (a *NDArray[T]) SubScalar(s T) *NDArray[T]
func (a *NDArray[T]) MulScalar(s T) *NDArray[T]
func (a *NDArray[T]) DivScalar(s T) *NDArray[T]
func (a *NDArray[T]) PowScalar(p float64) *NDArray[float64]

// Array ops (element-wise, requires same shape or broadcastable)
func Add[T constraints.Number](a, b *NDArray[T]) (*NDArray[T], error)
func Sub[T constraints.Number](a, b *NDArray[T]) (*NDArray[T], error)
func Mul[T constraints.Number](a, b *NDArray[T]) (*NDArray[T], error)
func Div[T constraints.Number](a, b *NDArray[T]) (*NDArray[T], error)

// Unary
func (a *NDArray[T]) Abs() *NDArray[T]
func (a *NDArray[T]) Neg() *NDArray[T]
func (a *NDArray[float64]) Sqrt() *NDArray[float64]
func (a *NDArray[float64]) Exp() *NDArray[float64]
func (a *NDArray[float64]) Log() *NDArray[float64]

// Reduction (axis=-1 means all axes collapsed to scalar)
func (a *NDArray[T]) Sum(axis int) *NDArray[T]
func (a *NDArray[T]) Min(axis int) *NDArray[T]
func (a *NDArray[T]) Max(axis int) *NDArray[T]
func (a *NDArray[float64]) Mean(axis int) *NDArray[float64]

// Comparison (returns *NDArray[bool])
func (a *NDArray[T]) Eq(b *NDArray[T]) (*NDArray[bool], error)
func (a *NDArray[T]) Gt(b *NDArray[T]) (*NDArray[bool], error)
func (a *NDArray[T]) Lt(b *NDArray[T]) (*NDArray[bool], error)
```

### 5.4 Linear Algebra (`ndarray/ndarray_linalg.go`)

```go
// Dot computes dot product of two 1D arrays (vectors).
func Dot[T constraints.Number](a, b *NDArray[T]) (T, error)

// MatMul computes matrix multiplication (2D arrays only).
func MatMul[T constraints.Number](a, b *NDArray[T]) (*NDArray[T], error)

// Inv computes matrix inverse (float64 only, 2D square).
func Inv(a *NDArray[float64]) (*NDArray[float64], error)

// Det computes determinant (float64 only, 2D square).
func Det(a *NDArray[float64]) (float64, error)

// SVD computes Singular Value Decomposition.
// Returns U, S, Vt such that a ≈ U · diag(S) · Vt
func SVD(a *NDArray[float64]) (U, S, Vt *NDArray[float64], err error)

// Norm computes L2 norm (vector) or Frobenius norm (matrix).
func Norm[T constraints.Number](a *NDArray[T]) float64

// Solve solves a · x = b for x.
func Solve(a, b *NDArray[float64]) (*NDArray[float64], error)
```

Broadcasting rules to implement:
- Align shapes from the right
- Dimensions of size 1 can be stretched to match
- Mismatch without a size-1 dimension → return error

---

## PHASE 6 — IO LAYER (`io/`)

### 6.1 CSV (`io/csv.go`)

```go
package io

type CSVReadOptions struct {
    Delimiter   rune              // default: ','
    HasHeader   bool              // default: true
    SkipRows    int               // skip N rows at top
    NullValues  []string          // strings treated as null (default: ["", "NA", "NaN", "null"])
    InferDtypes bool              // auto-detect int64/float64/bool (default: true)
    ChunkSize   int               // rows per chunk for streaming (0 = all at once)
    Columns     []string          // select specific columns only
    MaxRows     int               // stop after N rows (0 = unlimited)
}

func DefaultCSVReadOptions() CSVReadOptions

// ReadCSV reads a CSV file into a DataFrame.
func ReadCSV(path string, opts ...CSVReadOptions) (*dataframe.DataFrame, error)

// ReadCSVChunked returns a channel of DataFrames for memory-efficient processing.
func ReadCSVChunked(path string, opts CSVReadOptions) (<-chan *dataframe.DataFrame, <-chan error)

type CSVWriteOptions struct {
    Delimiter rune
    Header    bool
    NullAs    string    // how to write nulls (default: "")
}

// WriteCSV writes a DataFrame to a CSV file.
func WriteCSV(df *dataframe.DataFrame, path string, opts ...CSVWriteOptions) error
```

### 6.2 JSON (`io/json.go`)

```go
type JSONOrientation string

const (
    JSONOrientRecords JSONOrientation = "records"   // [{"col": val}, ...]
    JSONOrientColumns JSONOrientation = "columns"   // {"col": [val, ...], ...}
)

type JSONReadOptions struct {
    Orientation JSONOrientation
    NullValues  []string
}

func ReadJSON(path string, opts ...JSONReadOptions) (*dataframe.DataFrame, error)
func ReadJSONBytes(data []byte, opts ...JSONReadOptions) (*dataframe.DataFrame, error)
func WriteJSON(df *dataframe.DataFrame, path string, orientation JSONOrientation) error
func ToJSONBytes(df *dataframe.DataFrame, orientation JSONOrientation) ([]byte, error)
```

### 6.3 Parquet Interface Only (`io/parquet_iface.go`)

```go
// ParquetReader is the interface a Parquet backend must implement.
// We define the contract here; implementation is left to the caller
// (e.g. using github.com/parquet-go/parquet-go).
type ParquetReader interface {
    Read(path string) (*dataframe.DataFrame, error)
}

type ParquetWriter interface {
    Write(df *dataframe.DataFrame, path string) error
}

// RegisterParquetReader plugs in a third-party implementation.
func RegisterParquetReader(r ParquetReader)
func RegisterParquetWriter(w ParquetWriter)
```

---

## PHASE 7 — EXPRESSION DSL (`expr/expr.go`)

Provide a composable expression system for filtering — no string parsing required.

```go
package expr

// Expr is a composable boolean predicate over a DataFrame row.
type Expr interface {
    Eval(row map[string]any) bool
}

// Col starts an expression chain for a named column.
func Col(name string) *ColExpr

type ColExpr struct{ name string }

// Comparison ops
func (c *ColExpr) Eq(val any) Expr
func (c *ColExpr) Ne(val any) Expr
func (c *ColExpr) Gt(val any) Expr
func (c *ColExpr) Gte(val any) Expr
func (c *ColExpr) Lt(val any) Expr
func (c *ColExpr) Lte(val any) Expr
func (c *ColExpr) In(values ...any) Expr
func (c *ColExpr) IsNull() Expr
func (c *ColExpr) IsNotNull() Expr
func (c *ColExpr) Contains(sub string) Expr      // string only
func (c *ColExpr) StartsWith(prefix string) Expr
func (c *ColExpr) Matches(pattern string) Expr   // regex

// Boolean combinators
func And(exprs ...Expr) Expr
func Or(exprs ...Expr) Expr
func Not(e Expr) Expr

// Example usage:
//
//   df.FilterExpr(
//       expr.And(
//           expr.Col("age").Gte(30),
//           expr.Col("dept").In("Engineering", "Product"),
//           expr.Not(expr.Col("salary").IsNull()),
//       ),
//   )
```

---

## PHASE 8 — PARALLEL EXECUTION (`parallel/`)

### 8.1 Worker Pool (`parallel/pool.go`)

```go
package parallel

type Pool struct {
    workers int
}

// NewPool creates a pool. workers=0 uses runtime.NumCPU().
func NewPool(workers int) *Pool

func (p *Pool) Workers() int
```

### 8.2 Parallel Operations (`parallel/parallel_ops.go`)

```go
// Map applies fn concurrently to a Series, chunk-by-chunk.
// chunkSize=0 auto-computes based on series length and worker count.
func Map[T any](pool *Pool, s *series.Series[T], fn func(T) T, chunkSize int) *series.Series[T]

// Reduce applies fn in parallel using a tree-reduction pattern.
func Reduce[T any](pool *Pool, s *series.Series[T], initial T, fn func(a, b T) T) T

// Filter runs a predicate in parallel, preserving order.
func Filter[T any](pool *Pool, s *series.Series[T], fn func(T) bool) *series.Series[T]

// ForEach runs fn on each chunk concurrently with no return value.
func ForEach[T any](pool *Pool, s *series.Series[T], fn func(chunk []T))

// ParallelDF wraps a DataFrame for parallel column operations.
type ParallelDF struct {
    df   *dataframe.DataFrame
    pool *Pool
}

// Parallel returns a ParallelDF bound to df.
// Usage: df.Parallel().Apply(...)
func (df *dataframe.DataFrame) Parallel(pool ...*Pool) *ParallelDF

func (p *ParallelDF) Apply(column string, fn func(any, bool) (any, bool)) (*dataframe.DataFrame, error)
func (p *ParallelDF) FilterExpr(e expr.Expr) (*dataframe.DataFrame, error)
```

---

## PHASE 9 — STATISTICS (`stats/`)

### 9.1 Descriptive (`stats/descriptive.go`)

```go
package stats

// All functions accept []float64 for simplicity.
// Series-level equivalents live in series/series_stats.go.

func Mean(data []float64) float64
func Median(data []float64) float64
func Mode(data []float64) []float64      // can be multimodal
func Variance(data []float64) float64    // sample variance
func Std(data []float64) float64         // sample std dev
func Skewness(data []float64) float64
func Kurtosis(data []float64) float64
func Covariance(x, y []float64) float64
func Correlation(x, y []float64) (float64, error)   // Pearson r
func Quantile(data []float64, q float64) float64
func IQR(data []float64) float64
func ZScore(data []float64) []float64
func CumSum(data []float64) []float64
func CumProd(data []float64) []float64
func RollingMean(data []float64, window int) []float64
func RollingStd(data []float64, window int) []float64
```

### 9.2 Distributions (`stats/distributions.go`)

```go
// Each distribution implements PDF, CDF, PPF (inverse CDF), Sample.
type Distribution interface {
    PDF(x float64) float64
    CDF(x float64) float64
    PPF(p float64) float64           // inverse CDF / quantile function
    Sample(n int) []float64
    Mean() float64
    Variance() float64
}

func Normal(mean, std float64) Distribution
func Uniform(low, high float64) Distribution
func Binomial(n int, p float64) Distribution
func Poisson(lambda float64) Distribution
func Exponential(rate float64) Distribution
func TDist(df float64) Distribution
func ChiSquared(df float64) Distribution
```

---

## PHASE 10 — ERROR HANDLING RULES

Apply consistently across ALL packages:

```go
// CORRECT: explicit, descriptive errors
func (df *DataFrame) Col(name string) (column, error) {
    idx, ok := df.index[name]
    if !ok {
        return nil, fmt.Errorf("gods/dataframe: column %q not found (available: %v)",
            name, df.Columns())
    }
    return df.cols[idx], nil
}

// WRONG: panics, opaque errors, or silent failures
// WRONG: returning zero values without an error
// WRONG: strconv.ErrSyntax without context wrapping
```

Rules:
- No `panic` in any exported function (use `Must*` variants for caller-controlled panics)
- Wrap errors: `fmt.Errorf("gods/package: context: %w", err)`
- Never return `nil, nil` — either the result or the error must be non-nil
- Validate inputs at the boundary, not deep in call stacks
- `Must*` constructors (e.g. `MustAt`) are allowed but must document that they panic

---

## PHASE 11 — BENCHMARKS & TESTS

### 11.1 Test Requirements

Every package must have `_test.go` covering:

```
✓ Happy path with typical data
✓ Empty input (0 rows, 0 cols)
✓ Single element
✓ All-null column / all-valid column
✓ Very large input (≥ 1M rows for Series; gate behind testing.Short())
✓ Concurrent access (run with -race flag)
✓ Type boundary values (MaxInt64, NaN, Inf, -0.0)
```

### 11.2 Benchmark Template

```go
func BenchmarkSeriesMap(b *testing.B) {
    s := series.New("x", make([]float64, 1_000_000))
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        s.Map(func(v float64) float64 { return v * 2.0 })
    }
}

func BenchmarkDataFrameFilter(b *testing.B) {
    df := generateLargeDF(b, 1_000_000)
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        df.Filter(func(r dataframe.Row) bool {
            age, _ := r["age"].(int64)
            return age > 30
        })
    }
}
```

Run with:
```bash
go test -bench=. -benchmem -benchtime=5s ./...
go test -race ./...
```

---

## PHASE 12 — EXAMPLE USAGE (README.md)

```go
package main

import (
    "fmt"
    "log"

    "github.com/yourusername/gods/dataframe"
    "github.com/yourusername/gods/expr"
    gods_io "github.com/yourusername/gods/io"
    "github.com/yourusername/gods/parallel"
    "github.com/yourusername/gods/series"
    "github.com/yourusername/gods/stats"
)

func main() {
    // IO
    df, err := gods_io.ReadCSV("employees.csv")
    if err != nil { log.Fatal(err) }

    fmt.Println("Shape:", df.Shape())
    df.Info()

    // Filtering with expression DSL
    seniorEng, err := df.FilterExpr(
        expr.And(
            expr.Col("age").Gte(35),
            expr.Col("department").In("Engineering", "Data"),
            expr.Col("salary").IsNotNull(),
        ),
    )
    if err != nil { log.Fatal(err) }

    // GroupBy + Aggregation
    summary, err := df.GroupBy("department").Agg(map[string]dataframe.AggFunc{
        "salary": dataframe.AggMean,
        "age":    dataframe.AggCount,
    })
    if err != nil { log.Fatal(err) }
    fmt.Println(summary)

    // Parallel map on a large Series
    pool := parallel.NewPool(0) // all CPUs
    salaries := series.MustAs[float64](df.MustCol("salary"))
    inflated := parallel.Map(pool, salaries, func(v float64) float64 {
        return v * 1.10
    }, 0)
    _ = inflated

    // Statistics
    data := salaries.ValidValues()
    fmt.Printf("Mean: %.2f  Std: %.2f  P95: %.2f\n",
        stats.Mean(data),
        stats.Std(data),
        stats.Quantile(data, 0.95),
    )

    // Multi-key sort
    sorted, err := seniorEng.Sort(
        dataframe.SortKey{Column: "salary", Ascending: false},
        dataframe.SortKey{Column: "age", Ascending: true},
    )
    if err != nil { log.Fatal(err) }
    fmt.Println(sorted.Head(5))
}
```

---

## IMPLEMENTATION ORDER

Build in this exact sequence. Each phase must compile and pass tests before proceeding.

```
Phase 1  → Scaffold: go.mod, directory structure
Phase 2  → internal/bitmap + internal/util (no external deps)
Phase 3  → series (core type, access, null handling, stats)
Phase 4  → dataframe (core, filter, sort, groupby, join)
Phase 5  → ndarray (core, ops, linalg)
Phase 6  → io (csv first, then json, parquet interface)
Phase 7  → expr DSL
Phase 8  → parallel
Phase 9  → stats
Phase 10 → README + examples
Phase 11 → Benchmarks across all packages
```

---

## HARD CONSTRAINTS — NEVER VIOLATE

1. **No reflection in hot paths** — type-switch is acceptable; `reflect.Value` is not
2. **Columnar storage only** — DataFrames are never stored row-by-row internally
3. **Zero-copy slices where possible** — return views, not copies, when safe
4. **All errors surfaced** — no silent data loss or coercion without an error
5. **Go 1.21 generics** — use `constraints.Number` from `golang.org/x/exp/constraints`
6. **No global mutable state** — thread safety via struct-level sync where needed
7. **Godoc on every exported symbol** — no exceptions

---

## FUTURE EXTENSIONS (DO NOT IMPLEMENT NOW — DESIGN FOR THEM)

- Apache Arrow IPC memory format (columnar interchange)
- Lazy evaluation / query planner (Polars-style)
- SQL interface via `database/sql` driver
- GPU kernels via CGo + CUDA
- SIMD intrinsics via assembly for float64 ops
- Python interop via `os/exec` + Arrow
