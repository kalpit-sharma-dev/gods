package io

import "github.com/yourusername/gods/dataframe"

// ParquetReader is the interface a Parquet backend must implement.
type ParquetReader interface {
	// Read reads parquet data from path into a DataFrame.
	Read(path string) (*dataframe.DataFrame, error)
}

// ParquetWriter is the interface a Parquet backend must implement for writes.
type ParquetWriter interface {
	// Write writes a DataFrame to parquet path.
	Write(df *dataframe.DataFrame, path string) error
}

var (
	registeredParquetReader ParquetReader
	registeredParquetWriter ParquetWriter
)

// RegisterParquetReader registers a third-party parquet reader implementation.
func RegisterParquetReader(r ParquetReader) {
	registeredParquetReader = r
}

// RegisterParquetWriter registers a third-party parquet writer implementation.
func RegisterParquetWriter(w ParquetWriter) {
	registeredParquetWriter = w
}
