package dataframe

import (
	"math"
	"testing"
	"time"

	"github.com/kalpit-sharma-dev/gods/expr"
)

func TestDataFrameBasicOps(t *testing.T) {
	df, err := FromMap(map[string]any{
		"id":   []int64{1, 2, 3},
		"name": []string{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("FromMap error: %v", err)
	}
	rows, cols := df.Shape()
	if rows != 3 || cols != 2 {
		t.Fatalf("unexpected shape %d x %d", rows, cols)
	}
	row, err := df.Row(1)
	if err != nil {
		t.Fatalf("row error: %v", err)
	}
	if row["name"] != "b" {
		t.Fatalf("unexpected row value: %#v", row["name"])
	}
}

func TestDataFrameFilterExpr(t *testing.T) {
	df, err := FromMap(map[string]any{
		"age":  []int64{21, 35, 40},
		"dept": []string{"A", "B", "B"},
	})
	if err != nil {
		t.Fatalf("FromMap error: %v", err)
	}
	out, err := df.FilterExpr(expr.And(
		expr.Col("age").Gte(int64(30)),
		expr.Col("dept").Eq("B"),
	))
	if err != nil {
		t.Fatalf("FilterExpr error: %v", err)
	}
	rows, _ := out.Shape()
	if rows != 2 {
		t.Fatalf("expected 2 rows, got %d", rows)
	}
}

func TestGroupByAgg(t *testing.T) {
	df, err := FromMap(map[string]any{
		"dept":   []string{"A", "A", "B"},
		"salary": []float64{100, 200, 300},
	})
	if err != nil {
		t.Fatalf("FromMap error: %v", err)
	}
	gb, err := df.GroupBy("dept")
	if err != nil {
		t.Fatalf("GroupBy error: %v", err)
	}
	out, err := gb.Agg(map[string]AggFunc{"salary": AggMean})
	if err != nil {
		t.Fatalf("Agg error: %v", err)
	}
	rows, _ := out.Shape()
	if rows != 2 {
		t.Fatalf("expected 2 group rows, got %d", rows)
	}
}

func TestGroupByTypedKeysRemainTyped(t *testing.T) {
	df, err := FromMap(map[string]any{
		"k": []int64{1, 1, 2},
		"v": []float64{10, 20, 30},
	})
	if err != nil {
		t.Fatalf("FromMap error: %v", err)
	}
	gb, err := df.GroupBy("k")
	if err != nil {
		t.Fatalf("GroupBy error: %v", err)
	}
	out, err := gb.Agg(map[string]AggFunc{"v": AggSum})
	if err != nil {
		t.Fatalf("Agg error: %v", err)
	}
	r0, err := out.Row(0)
	if err != nil {
		t.Fatalf("row error: %v", err)
	}
	if _, ok := r0["k"].(int64); !ok {
		t.Fatalf("expected grouped key type int64, got %T", r0["k"])
	}
}

func TestSortPreservesColumnOrder(t *testing.T) {
	df, err := FromMap(map[string]any{
		"b": []int64{2, 1},
		"a": []string{"x", "y"},
		"c": []float64{1.1, 2.2},
	})
	if err != nil {
		t.Fatalf("FromMap error: %v", err)
	}
	out, err := df.Sort(SortKey{Column: "b", Ascending: true})
	if err != nil {
		t.Fatalf("Sort error: %v", err)
	}
	want := []string{"a", "b", "c"}
	got := out.Columns()
	if len(got) != len(want) {
		t.Fatalf("unexpected columns length: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("column order mismatch at %d: got %q want %q", i, got[i], want[i])
		}
	}
}

func TestJoinKeyCollisionResistance(t *testing.T) {
	left, err := FromMap(map[string]any{
		"k1": []string{"a||b", "a"},
		"k2": []string{"c", "b||c"},
		"v1": []int64{10, 20},
	})
	if err != nil {
		t.Fatalf("left FromMap error: %v", err)
	}
	right, err := FromMap(map[string]any{
		"k1": []string{"a||b", "a"},
		"k2": []string{"c", "b||c"},
		"v2": []int64{100, 200},
	})
	if err != nil {
		t.Fatalf("right FromMap error: %v", err)
	}
	out, err := Join(left, right, []string{"k1", "k2"}, InnerJoin, [2]string{"", ""})
	if err != nil {
		t.Fatalf("Join error: %v", err)
	}
	rows, _ := out.Shape()
	if rows != 2 {
		t.Fatalf("expected 2 joined rows, got %d", rows)
	}
	r0, _ := out.Row(0)
	r1, _ := out.Row(1)
	if r0["v1"] == r1["v1"] || r0["v2"] == r1["v2"] {
		t.Fatalf("unexpected duplicate pairing, possible key collision: r0=%v r1=%v", r0, r1)
	}
}

func TestRankDenseMonotonic(t *testing.T) {
	df, err := FromMap(map[string]any{
		"x": []float64{10, 10, 20, 30},
	})
	if err != nil {
		t.Fatalf("FromMap error: %v", err)
	}
	r, err := df.Rank("x", "dense")
	if err != nil {
		t.Fatalf("Rank error: %v", err)
	}
	want := []float64{1, 1, 2, 3}
	for i, w := range want {
		v, ok := r.At(i)
		if !ok {
			t.Fatalf("rank[%d] is null", i)
		}
		if math.Abs(v-w) > 1e-9 {
			t.Fatalf("rank[%d]=%v want %v", i, v, w)
		}
	}
}

func TestCompareValuesLargeInt64Precision(t *testing.T) {
	// Values differ by 1 but cannot be represented distinctly in float64.
	a := int64(9_223_372_036_854_775_000)
	b := int64(9_223_372_036_854_774_999)
	if compareValues(a, b) <= 0 {
		t.Fatalf("expected a>b for large int64 values, got compare=%d", compareValues(a, b))
	}
}

func TestJoinOnTimeKeys(t *testing.T) {
	t1 := time.Date(2026, 5, 2, 0, 0, 0, 0, time.UTC)
	t2 := t1.Add(24 * time.Hour)
	left, err := FromMap(map[string]any{
		"ts": []time.Time{t1, t2},
		"v1": []int64{1, 2},
	})
	if err != nil {
		t.Fatalf("left FromMap error: %v", err)
	}
	right, err := FromMap(map[string]any{
		"ts": []time.Time{t2, t1},
		"v2": []int64{20, 10},
	})
	if err != nil {
		t.Fatalf("right FromMap error: %v", err)
	}
	out, err := Join(left, right, []string{"ts"}, InnerJoin, [2]string{"", ""})
	if err != nil {
		t.Fatalf("Join error: %v", err)
	}
	rows, _ := out.Shape()
	if rows != 2 {
		t.Fatalf("expected 2 rows, got %d", rows)
	}
	r0, _ := out.Row(0)
	if _, ok := r0["ts"].(time.Time); !ok {
		t.Fatalf("expected time.Time join key type, got %T", r0["ts"])
	}
}
