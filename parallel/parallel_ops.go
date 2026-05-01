package parallel

import (
	"fmt"
	"sync"

	"github.com/yourusername/gods/dataframe"
	"github.com/yourusername/gods/expr"
	"github.com/yourusername/gods/series"
)

// Map applies fn concurrently to a Series, chunk-by-chunk.
func Map[T any](pool *Pool, s *series.Series[T], fn func(T) T, chunkSize int) *series.Series[T] {
	if s == nil {
		return nil
	}
	if fn == nil {
		return series.MustAs[T](s.CloneAny())
	}
	n := s.Len()
	if n == 0 {
		return series.New[T](s.Name(), nil)
	}
	p := ensurePool(pool)
	if chunkSize <= 0 {
		chunkSize = defaultChunkSize(n, p.Workers())
	}
	ranges := chunkRanges(n, chunkSize)
	outVals := make([]T, n)
	nullMask := make([]bool, n)
	var wg sync.WaitGroup
	jobs := make(chan [2]int)
	for w := 0; w < p.Workers(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := range jobs {
				for i := r[0]; i < r[1]; i++ {
					v, ok := s.At(i)
					if !ok {
						nullMask[i] = true
						continue
					}
					outVals[i] = fn(v)
				}
			}
		}()
	}
	for _, r := range ranges {
		jobs <- r
	}
	close(jobs)
	wg.Wait()
	return series.WithNulls(s.Name(), outVals, nullMask)
}

// Reduce applies fn in parallel using chunked partial reductions.
func Reduce[T any](pool *Pool, s *series.Series[T], initial T, fn func(a, b T) T) T {
	if s == nil || fn == nil || s.Len() == 0 {
		return initial
	}
	p := ensurePool(pool)
	n := s.Len()
	ranges := chunkRanges(n, defaultChunkSize(n, p.Workers()))
	partials := make([]T, len(ranges))
	var wg sync.WaitGroup
	for idx, r := range ranges {
		wg.Add(1)
		go func(outIdx int, start, end int) {
			defer wg.Done()
			acc := initial
			for i := start; i < end; i++ {
				v, ok := s.At(i)
				if !ok {
					continue
				}
				acc = fn(acc, v)
			}
			partials[outIdx] = acc
		}(idx, r[0], r[1])
	}
	wg.Wait()
	acc := initial
	for _, pval := range partials {
		acc = fn(acc, pval)
	}
	return acc
}

// Filter runs a predicate in parallel while preserving order.
func Filter[T any](pool *Pool, s *series.Series[T], fn func(T) bool) *series.Series[T] {
	if s == nil {
		return nil
	}
	if fn == nil {
		return series.New[T](s.Name(), nil)
	}
	p := ensurePool(pool)
	n := s.Len()
	keep := make([]bool, n)
	ranges := chunkRanges(n, defaultChunkSize(n, p.Workers()))
	var wg sync.WaitGroup
	for _, r := range ranges {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				v, ok := s.At(i)
				keep[i] = ok && fn(v)
			}
		}(r[0], r[1])
	}
	wg.Wait()
	values := make([]T, 0, n)
	for i := 0; i < n; i++ {
		if !keep[i] {
			continue
		}
		v, _ := s.At(i)
		values = append(values, v)
	}
	return series.New(s.Name(), values)
}

// ForEach runs fn on each chunk concurrently.
func ForEach[T any](pool *Pool, s *series.Series[T], fn func(chunk []T)) {
	if s == nil || fn == nil {
		return
	}
	values := s.Values()
	p := ensurePool(pool)
	ranges := chunkRanges(len(values), defaultChunkSize(len(values), p.Workers()))
	var wg sync.WaitGroup
	for _, r := range ranges {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			chunk := make([]T, end-start)
			copy(chunk, values[start:end])
			fn(chunk)
		}(r[0], r[1])
	}
	wg.Wait()
}

// ParallelDF wraps a DataFrame for parallel operations.
type ParallelDF struct {
	df   *dataframe.DataFrame
	pool *Pool
}

// Parallel binds a DataFrame to a parallel execution context.
func Parallel(df *dataframe.DataFrame, pool ...*Pool) *ParallelDF {
	var p *Pool
	if len(pool) > 0 {
		p = pool[0]
	}
	return &ParallelDF{df: df, pool: ensurePool(p)}
}

// Apply applies a DataFrame column function.
func (p *ParallelDF) Apply(column string, fn func(any, bool) (any, bool)) (*dataframe.DataFrame, error) {
	if p == nil || p.df == nil {
		return nil, fmt.Errorf("gods/parallel: nil dataframe")
	}
	_ = p.pool
	return p.df.Apply(column, fn)
}

// FilterExpr filters a DataFrame with an expression.
func (p *ParallelDF) FilterExpr(e expr.Expr) (*dataframe.DataFrame, error) {
	if p == nil || p.df == nil {
		return nil, fmt.Errorf("gods/parallel: nil dataframe")
	}
	_ = p.pool
	return p.df.FilterExpr(e)
}

func ensurePool(p *Pool) *Pool {
	if p == nil {
		return NewPool(0)
	}
	return p
}

func defaultChunkSize(n, workers int) int {
	if workers <= 0 {
		workers = 1
	}
	size := n / workers
	if size <= 0 {
		size = 1
	}
	return size
}

func chunkRanges(n, chunkSize int) [][2]int {
	if chunkSize <= 0 {
		chunkSize = 1
	}
	ranges := make([][2]int, 0, (n+chunkSize-1)/chunkSize)
	for start := 0; start < n; start += chunkSize {
		end := start + chunkSize
		if end > n {
			end = n
		}
		ranges = append(ranges, [2]int{start, end})
	}
	return ranges
}
