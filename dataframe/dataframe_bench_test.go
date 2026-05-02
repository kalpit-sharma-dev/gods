package dataframe

import "testing"

func benchDataFrame(rows int) *DataFrame {
	age := make([]int64, rows)
	dept := make([]string, rows)
	salary := make([]float64, rows)
	depts := []string{"Eng", "Data", "Product", "Sales"}
	for i := 0; i < rows; i++ {
		age[i] = int64((i % 70) + 18)
		dept[i] = depts[i%len(depts)]
		salary[i] = float64(50_000 + (i % 200_000))
	}
	df, _ := FromMap(map[string]any{
		"age":    age,
		"dept":   dept,
		"salary": salary,
	})
	return df
}

func BenchmarkDataFrameFilter(b *testing.B) {
	df := benchDataFrame(1_000_000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = df.Filter(func(r Row) bool {
			age, _ := r["age"].(int64)
			return age > 30
		})
	}
}

func BenchmarkDataFrameGroupByAgg(b *testing.B) {
	df := benchDataFrame(500_000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gb, _ := df.GroupBy("dept")
		_, _ = gb.Agg(map[string]AggFunc{
			"salary": AggMean,
			"age":    AggCount,
		})
	}
}

func BenchmarkDataFrameJoin(b *testing.B) {
	left := benchDataFrame(200_000)
	right := benchDataFrame(200_000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Join(left, right, []string{"age", "dept"}, InnerJoin, [2]string{"_l", "_r"})
	}
}
