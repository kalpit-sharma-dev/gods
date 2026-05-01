package dataframe

import (
	"testing"

	"github.com/yourusername/gods/expr"
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
