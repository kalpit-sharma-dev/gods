package parallel

import (
	"sync"
	"testing"

	"github.com/kalpit-sharma-dev/gods/dataframe"
	"github.com/kalpit-sharma-dev/gods/expr"
	"github.com/kalpit-sharma-dev/gods/series"
)

func TestMapReduceFilter(t *testing.T) {
	pool := NewPool(2)
	s := series.New("x", []int64{1, 2, 3, 4, 5})
	mapped := Map(pool, s, func(v int64) int64 { return v * 2 }, 0)
	if got := mapped.MustAt(2); got != 6 {
		t.Fatalf("unexpected mapped value: %d", got)
	}
	sum := Reduce(pool, mapped, int64(0), func(a, b int64) int64 { return a + b })
	if sum != 30 {
		t.Fatalf("unexpected reduce result: %d", sum)
	}
	filtered := Filter(pool, mapped, func(v int64) bool { return v > 5 })
	if filtered.Len() != 3 {
		t.Fatalf("unexpected filtered length: %d", filtered.Len())
	}
}

func TestParallelDF(t *testing.T) {
	df, err := dataframe.FromMap(map[string]any{
		"age":  []int64{20, 30, 40},
		"name": []string{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("FromMap failed: %v", err)
	}
	pdf := Parallel(df, NewPool(2))
	applied, err := pdf.Apply("age", func(v any, ok bool) (any, bool) {
		if !ok {
			return nil, false
		}
		return v.(int64) + 1, true
	})
	if err != nil {
		t.Fatalf("Parallel Apply failed: %v", err)
	}
	col, err := dataframe.SeriesAt[int64](applied, "age")
	if err != nil {
		t.Fatalf("SeriesAt failed: %v", err)
	}
	if col.MustAt(0) != 21 {
		t.Fatalf("unexpected apply output")
	}
	filtered, err := pdf.FilterExpr(expr.Col("age").Gt(int64(25)))
	if err != nil {
		t.Fatalf("FilterExpr failed: %v", err)
	}
	rows, _ := filtered.Shape()
	if rows != 2 {
		t.Fatalf("unexpected filtered rows: %d", rows)
	}
}

func TestMapConcurrencyRaceSafety(t *testing.T) {
	pool := NewPool(8)
	s := series.New("x", make([]int64, 100_000))
	for i := 0; i < s.Len(); i++ {
		s.Set(i, int64(i))
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out := Map(pool, s, func(v int64) int64 { return v + 1 }, 0)
			if out.Len() != s.Len() {
				t.Errorf("unexpected mapped len: got %d want %d", out.Len(), s.Len())
			}
		}()
	}
	wg.Wait()
}
