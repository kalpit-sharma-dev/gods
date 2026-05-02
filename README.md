# gods

`gods` is an idiomatic Go data manipulation and numerical computing library inspired by pandas and NumPy.

## Features

- Nullable, typed `Series` with bitmap-backed validity
- Columnar `DataFrame` with filtering, sorting, grouping, joins, and transforms
- Generic `NDArray` with broadcasting, indexing, reductions, and linear algebra helpers
- CSV/JSON IO layer plus pluggable parquet interfaces
- Expression DSL for composable predicates
- Parallel helpers for `Series` and `DataFrame` operations
- Descriptive statistics and common distributions

## Install

```bash
go get github.com/kalpit-sharma-dev/gods
```

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/kalpit-sharma-dev/gods/dataframe"
	"github.com/kalpit-sharma-dev/gods/expr"
	godsio "github.com/kalpit-sharma-dev/gods/io"
	"github.com/kalpit-sharma-dev/gods/parallel"
	"github.com/kalpit-sharma-dev/gods/series"
	"github.com/kalpit-sharma-dev/gods/stats"
)

func main() {
	df, err := godsio.ReadCSV("employees.csv")
	if err != nil {
		log.Fatal(err)
	}

	highEarners, err := df.FilterExpr(
		expr.And(
			expr.Col("department").Eq("Engineering"),
			expr.Col("salary").Gt(120000),
		),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(highEarners.Head(5))

	pool := parallel.NewPool(0)
	salary := dataframe.MustSeriesAt[float64](df, "salary")
	raised := parallel.Map(pool, salary, func(v float64) float64 { return v * 1.10 }, 0)
	fmt.Println(raised.Head(3))

	fmt.Printf("salary mean: %.2f\n", stats.Mean(salary.ValidValues()))

	_ = series.New("example", []int64{1, 2, 3})
}
```

## Development

```bash
go test ./...
```
