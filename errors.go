package ml

import "errors"

var (
	//General error to prevent panic
	ErrOutOfRange = errors.New("Index out of range")

	//Vector Errors
	//Error for vectors calculation operators
	//some operations need exact same dimension to work
	ErrVectorFalseDimension = errors.New("Dimensions of both vectors don't agree")

	//Matrix Errors
	//Error for matrix operations that require more than 1 vector
	//and the vectors should have the exact same dimension
	ErrMatrixIncorrentVectorLength = errors.New("Length of input vectors should be the same")
	//Error for empty matrix
	//some operations require it to have at least 1 row
	ErrEmptyMatrix = errors.New("Matrix is empty")
)
