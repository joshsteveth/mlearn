package ml

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

//all strings should be parseable into float64
func convertCSVToFloat64(rec [][]string) ([][]float64, error) {
	var result [][]float64
	for _, str := range rec {
		//str is a []string
		//create a new []float64 as a container for result
		var flt []float64
		for _, s := range str {
			//s is a string
			//which should be into float64 convertable
			//return error otherwise
			if f, err := strconv.ParseFloat(s, 64); err != nil {
				return nil, err
			} else {
				flt = append(flt, f)
			}
		}
		//append the []float64 into result
		result = append(result, flt)
	}

	return result, nil
}

//use row/col ":" if all data should be fetched
//otherwise either using 1 for e.g. row 1 only
//or 1:2 to fetch row 1-2
func vectorCat(cat string) ([]int, error) {
	//first case is check whether cat is == ":"
	//if that's the case then simply return empty slice and no error
	if cat == ":" {
		return []int{}, nil
	}

	//second case is to determine whether cat is a range or single value
	//split with ":", if len is 1 then single value, it's a range if 2, and error otherwise
	splitted := strings.Split(cat, ":")
	switch len(splitted) {
	case 1:
		//the result should be into int convertable
		if res, err := strconv.Atoi(splitted[0]); err != nil {
			return nil, err
		} else {
			if res < 1 {
				return nil, ErrOutOfRange
			}

			return []int{res}, nil
		}
	case 2:
		//now both of the value should be into int convertable
		var result []int
		for _, s := range splitted {
			if res, err := strconv.Atoi(s); err != nil {
				return nil, err
			} else {
				if res < 1 {
					return nil, ErrOutOfRange
				}

				result = append(result, res)
			}
		}

		//also return error if result[0] is not lesser than result[1]
		if result[0] >= result[1] {
			return nil, fmt.Errorf("Upper limit should be greater than lower")
		}

		return result, nil
	}
	//if the result is neither 1 nor 2 then return error
	return nil, fmt.Errorf("Invalid category syntax")
}

//select particular row and column from a [][]float64
//validate input first, return error if failed
func filterInputByCat(input [][]float64, row, col string) ([][]float64, error) {
	if err := validateMatrixInput(input); err != nil {
		return nil, err
	}

	//get the desired rows and cols
	//return if there's any error
	//get all result if rows/cols is an empty slice
	rows, err := vectorCat(row)
	if err != nil {
		return nil, err
	}

	cols, err := vectorCat(col)
	if err != nil {
		return nil, err
	}

	//first get the rows
	rowsV := [][]float64{}
	switch len(rows) {
	case 0:
		rowsV = input
	case 1:
		//validate the input to prevent panic
		if len(input) < rows[0] {
			return nil, ErrOutOfRange
		}
		//append only the selected row
		rowsV = append(rowsV, input[rows[0]-1])
	case 2:
		for k, r := range input {
			if (k+1) >= rows[0] && (k+1) <= rows[1] {
				rowsV = append(rowsV, r)
			}
		}
	}

	//now filter the columns
	result := [][]float64{}
	switch len(cols) {
	case 0:
		result = rowsV
	case 1:
		//validate the input to prevent panic
		if len(rowsV) == 0 || len(rowsV[0]) < cols[0] {
			return nil, ErrOutOfRange
		}
		//append only the selected row
		for _, flt := range rowsV {
			result = append(result, []float64{flt[cols[0]-1]})
		}
	case 2:
		for _, flt := range rowsV {
			var f []float64
			for k, c := range flt {
				if (k+1) >= cols[0] && (k+1) <= cols[1] {
					f = append(f, c)
				}
			}
			result = append(result, f)
		}
	}

	return result, nil
}

//function to call the sigmoid func
//sigmoid is defined as g(z) = 1 / (1 + e^(-z))
func sigm(z float64) float64 {
	return 1 / (1 + math.Pow(math.E, -1*z))
}
