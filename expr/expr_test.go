package expr

import "testing"

func TestExprCombinators(t *testing.T) {
	row := map[string]any{
		"age":   36,
		"dept":  "Engineering",
		"alive": true,
	}
	e := And(
		Col("age").Gte(30),
		Col("dept").In("Engineering", "Data"),
		Not(Col("alive").Eq(false)),
	)
	if !e.Eval(row) {
		t.Fatalf("expected expression to match row")
	}
	if Col("dept").StartsWith("Sales").Eval(row) {
		t.Fatalf("unexpected startswith match")
	}
}
