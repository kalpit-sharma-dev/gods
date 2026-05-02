package io

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"

	"github.com/kalpit-sharma-dev/gods/dataframe"
	"github.com/kalpit-sharma-dev/gods/series"
)

// CSVReadOptions configures CSV reading behavior.
type CSVReadOptions struct {
	Delimiter   rune
	HasHeader   bool
	SkipRows    int
	NullValues  []string
	KeepNaN     bool
	InferDtypes bool
	ChunkSize   int
	Columns     []string
	MaxRows     int
}

// DefaultCSVReadOptions returns default options for CSV reads.
func DefaultCSVReadOptions() CSVReadOptions {
	return CSVReadOptions{
		Delimiter:   ',',
		HasHeader:   true,
		NullValues:  []string{"", "NA", "NaN", "null"},
		KeepNaN:     false,
		InferDtypes: true,
	}
}

// CSVWriteOptions configures CSV writing behavior.
type CSVWriteOptions struct {
	Delimiter rune
	Header    bool
	NullAs    string
}

// ReadCSV reads a CSV file into a DataFrame.
func ReadCSV(path string, opts ...CSVReadOptions) (*dataframe.DataFrame, error) {
	options := DefaultCSVReadOptions()
	if len(opts) > 0 {
		options = mergeReadOptions(options, opts[0])
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gods/io: open csv %q: %w", path, err)
	}
	defer file.Close()

	r := csv.NewReader(file)
	r.Comma = options.Delimiter
	records, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("gods/io: read csv %q: %w", path, err)
	}
	return recordsToDataFrame(records, options)
}

// ReadCSVChunked reads CSV and streams DataFrame chunks.
func ReadCSVChunked(path string, opts CSVReadOptions) (<-chan *dataframe.DataFrame, <-chan error) {
	out := make(chan *dataframe.DataFrame)
	errCh := make(chan error, 1)

	go func() {
		defer close(out)
		defer close(errCh)
		if opts.ChunkSize <= 0 {
			df, err := ReadCSV(path, opts)
			if err != nil {
				errCh <- err
				return
			}
			out <- df
			return
		}

		df, err := ReadCSV(path, opts)
		if err != nil {
			errCh <- err
			return
		}
		rows, _ := df.Shape()
		for start := 0; start < rows; start += opts.ChunkSize {
			end := start + opts.ChunkSize
			if end > rows {
				end = rows
			}
			chunkRows := make([]map[string]any, 0, end-start)
			for i := start; i < end; i++ {
				row, err := df.Row(i)
				if err != nil {
					errCh <- fmt.Errorf("gods/io: row %d: %w", i, err)
					return
				}
				chunkRows = append(chunkRows, row)
			}
			chunk, err := rowsToDataFrame(chunkRows)
			if err != nil {
				errCh <- err
				return
			}
			out <- chunk
		}
	}()

	return out, errCh
}

// WriteCSV writes a DataFrame to a CSV file.
func WriteCSV(df *dataframe.DataFrame, path string, opts ...CSVWriteOptions) error {
	if df == nil {
		return fmt.Errorf("gods/io: nil dataframe")
	}
	options := CSVWriteOptions{
		Delimiter: ',',
		Header:    true,
		NullAs:    "",
	}
	if len(opts) > 0 {
		if opts[0].Delimiter != 0 {
			options.Delimiter = opts[0].Delimiter
		}
		options.Header = opts[0].Header
		options.NullAs = opts[0].NullAs
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("gods/io: create csv %q: %w", path, err)
	}
	defer file.Close()

	w := csv.NewWriter(file)
	w.Comma = options.Delimiter
	cols := df.Columns()
	if options.Header {
		if err := w.Write(cols); err != nil {
			return fmt.Errorf("gods/io: write header: %w", err)
		}
	}
	rows, _ := df.Shape()
	for i := 0; i < rows; i++ {
		row, err := df.Row(i)
		if err != nil {
			return fmt.Errorf("gods/io: row %d: %w", i, err)
		}
		record := make([]string, len(cols))
		for j, c := range cols {
			v := row[c]
			if v == nil {
				record[j] = options.NullAs
			} else {
				record[j] = fmt.Sprint(v)
			}
		}
		if err := w.Write(record); err != nil {
			return fmt.Errorf("gods/io: write row %d: %w", i, err)
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return fmt.Errorf("gods/io: flush csv writer: %w", err)
	}
	return nil
}

func recordsToDataFrame(records [][]string, opts CSVReadOptions) (*dataframe.DataFrame, error) {
	if opts.SkipRows > 0 {
		if opts.SkipRows >= len(records) {
			return dataframe.Empty(), nil
		}
		records = records[opts.SkipRows:]
	}
	if len(records) == 0 {
		return dataframe.Empty(), nil
	}

	var headers []string
	startRow := 0
	if opts.HasHeader {
		headers = append([]string(nil), records[0]...)
		startRow = 1
	} else {
		headers = make([]string, len(records[0]))
		for i := range headers {
			headers[i] = fmt.Sprintf("col_%d", i)
		}
	}

	if opts.MaxRows > 0 && startRow+opts.MaxRows < len(records) {
		records = records[:startRow+opts.MaxRows]
	}

	keepSet := map[string]struct{}{}
	if len(opts.Columns) > 0 {
		for _, c := range opts.Columns {
			keepSet[c] = struct{}{}
		}
	}
	selectedIdx := make([]int, 0, len(headers))
	selectedHeaders := make([]string, 0, len(headers))
	for i, h := range headers {
		if len(keepSet) > 0 {
			if _, ok := keepSet[h]; !ok {
				continue
			}
		}
		selectedIdx = append(selectedIdx, i)
		selectedHeaders = append(selectedHeaders, h)
	}
	if len(selectedHeaders) == 0 {
		return dataframe.Empty(), nil
	}

	colValues := make(map[string][]string, len(selectedHeaders))
	for _, h := range selectedHeaders {
		colValues[h] = make([]string, 0, len(records)-startRow)
	}
	for r := startRow; r < len(records); r++ {
		row := records[r]
		for k, idx := range selectedIdx {
			h := selectedHeaders[k]
			if idx < len(row) {
				colValues[h] = append(colValues[h], row[idx])
			} else {
				colValues[h] = append(colValues[h], "")
			}
		}
	}

	out := map[string]any{}
	for _, h := range selectedHeaders {
		values := colValues[h]
		nullMask := make([]bool, len(values))
		for i, v := range values {
			if isNullValue(v, opts.NullValues, opts.KeepNaN) {
				nullMask[i] = true
			}
		}
		col, err := inferCSVSeries(h, values, nullMask, opts.InferDtypes)
		if err != nil {
			return nil, err
		}
		out[h] = col
	}
	return dataframe.FromMap(out)
}

func inferCSVSeries(name string, values []string, nullMask []bool, infer bool) (any, error) {
	if !infer {
		return series.WithNulls(name, values, nullMask), nil
	}
	if asInt, ok := parseAllInt64(values, nullMask); ok {
		return series.WithNulls(name, asInt, nullMask), nil
	}
	if asFloat, ok := parseAllFloat64(values, nullMask); ok {
		return series.WithNulls(name, asFloat, nullMask), nil
	}
	if asBool, ok := parseAllBool(values, nullMask); ok {
		return series.WithNulls(name, asBool, nullMask), nil
	}
	return series.WithNulls(name, values, nullMask), nil
}

func parseAllInt64(values []string, nullMask []bool) ([]int64, bool) {
	out := make([]int64, len(values))
	for i, v := range values {
		if i < len(nullMask) && nullMask[i] {
			continue
		}
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return nil, false
		}
		out[i] = n
	}
	return out, true
}

func parseAllFloat64(values []string, nullMask []bool) ([]float64, bool) {
	out := make([]float64, len(values))
	for i, v := range values {
		if i < len(nullMask) && nullMask[i] {
			continue
		}
		n, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, false
		}
		out[i] = n
	}
	return out, true
}

func parseAllBool(values []string, nullMask []bool) ([]bool, bool) {
	out := make([]bool, len(values))
	for i, v := range values {
		if i < len(nullMask) && nullMask[i] {
			continue
		}
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, false
		}
		out[i] = b
	}
	return out, true
}

func isNullValue(v string, nullValues []string, keepNaN bool) bool {
	for _, nv := range nullValues {
		if keepNaN && (v == "NaN" || v == "nan" || v == "NAN") && (nv == "NaN" || nv == "nan" || nv == "NAN") {
			continue
		}
		if v == nv {
			return true
		}
	}
	return false
}

func mergeReadOptions(base, in CSVReadOptions) CSVReadOptions {
	if in.Delimiter != 0 {
		base.Delimiter = in.Delimiter
	}
	base.HasHeader = in.HasHeader
	base.SkipRows = in.SkipRows
	if len(in.NullValues) > 0 {
		base.NullValues = append([]string(nil), in.NullValues...)
	}
	base.KeepNaN = in.KeepNaN
	base.InferDtypes = in.InferDtypes
	base.ChunkSize = in.ChunkSize
	base.MaxRows = in.MaxRows
	if len(in.Columns) > 0 {
		base.Columns = append([]string(nil), in.Columns...)
	}
	return base
}

func rowsToDataFrame(rows []map[string]any) (*dataframe.DataFrame, error) {
	if len(rows) == 0 {
		return dataframe.Empty(), nil
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
	data := map[string]any{}
	for _, c := range cols {
		values := make([]any, len(rows))
		nullMask := make([]bool, len(rows))
		for i, row := range rows {
			v, ok := row[c]
			if !ok || v == nil {
				nullMask[i] = true
				continue
			}
			values[i] = v
		}
		data[c] = series.WithNulls(c, values, nullMask)
	}
	return dataframe.FromMap(data)
}
