package io

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/kalpit-sharma-dev/gods/dataframe"
	"github.com/kalpit-sharma-dev/gods/series"
)

// JSONOrientation controls JSON table representation.
type JSONOrientation string

const (
	// JSONOrientRecords stores data as [{"col": val}, ...].
	JSONOrientRecords JSONOrientation = "records"
	// JSONOrientColumns stores data as {"col":[val,...], ...}.
	JSONOrientColumns JSONOrientation = "columns"
)

// JSONReadOptions configures JSON reading behavior.
type JSONReadOptions struct {
	Orientation JSONOrientation
	NullValues  []string
}

// ReadJSON reads a JSON file into a DataFrame.
func ReadJSON(path string, opts ...JSONReadOptions) (*dataframe.DataFrame, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("gods/io: read json %q: %w", path, err)
	}
	return ReadJSONBytes(data, opts...)
}

// ReadJSONBytes reads JSON bytes into a DataFrame.
func ReadJSONBytes(data []byte, opts ...JSONReadOptions) (*dataframe.DataFrame, error) {
	options := JSONReadOptions{Orientation: JSONOrientRecords}
	if len(opts) > 0 {
		options = opts[0]
		if options.Orientation == "" {
			options.Orientation = JSONOrientRecords
		}
	}
	switch options.Orientation {
	case JSONOrientRecords:
		var rows []map[string]any
		if err := json.Unmarshal(data, &rows); err != nil {
			return nil, fmt.Errorf("gods/io: parse records json: %w", err)
		}
		return rowsToDataFrame(rows)
	case JSONOrientColumns:
		var cols map[string][]any
		if err := json.Unmarshal(data, &cols); err != nil {
			return nil, fmt.Errorf("gods/io: parse columns json: %w", err)
		}
		return columnsToDataFrame(cols)
	default:
		return nil, fmt.Errorf("gods/io: unsupported json orientation %q", options.Orientation)
	}
}

// WriteJSON writes a DataFrame to a JSON file.
func WriteJSON(df *dataframe.DataFrame, path string, orientation JSONOrientation) error {
	data, err := ToJSONBytes(df, orientation)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("gods/io: write json %q: %w", path, err)
	}
	return nil
}

// ToJSONBytes serializes a DataFrame into JSON bytes.
func ToJSONBytes(df *dataframe.DataFrame, orientation JSONOrientation) ([]byte, error) {
	if df == nil {
		return nil, fmt.Errorf("gods/io: nil dataframe")
	}
	if orientation == "" {
		orientation = JSONOrientRecords
	}
	rows, _ := df.Shape()
	switch orientation {
	case JSONOrientRecords:
		out := make([]map[string]any, 0, rows)
		for i := 0; i < rows; i++ {
			row, err := df.Row(i)
			if err != nil {
				return nil, fmt.Errorf("gods/io: row %d: %w", i, err)
			}
			out = append(out, row)
		}
		data, err := json.Marshal(out)
		if err != nil {
			return nil, fmt.Errorf("gods/io: marshal records json: %w", err)
		}
		return data, nil
	case JSONOrientColumns:
		colNames := df.Columns()
		out := map[string][]any{}
		for _, c := range colNames {
			out[c] = make([]any, rows)
		}
		for i := 0; i < rows; i++ {
			row, err := df.Row(i)
			if err != nil {
				return nil, fmt.Errorf("gods/io: row %d: %w", i, err)
			}
			for _, c := range colNames {
				out[c][i] = row[c]
			}
		}
		data, err := json.Marshal(out)
		if err != nil {
			return nil, fmt.Errorf("gods/io: marshal columns json: %w", err)
		}
		return data, nil
	default:
		return nil, fmt.Errorf("gods/io: unsupported json orientation %q", orientation)
	}
}

func columnsToDataFrame(cols map[string][]any) (*dataframe.DataFrame, error) {
	if len(cols) == 0 {
		return dataframe.Empty(), nil
	}
	maxLen := 0
	for _, values := range cols {
		if len(values) > maxLen {
			maxLen = len(values)
		}
	}
	data := map[string]any{}
	for name, values := range cols {
		padded := make([]any, maxLen)
		nullMask := make([]bool, maxLen)
		copy(padded, values)
		for i := 0; i < maxLen; i++ {
			if i >= len(values) || values[i] == nil {
				nullMask[i] = true
			}
		}
		data[name] = series.WithNulls(name, padded, nullMask)
	}
	return dataframe.FromMap(data)
}
