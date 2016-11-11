package ml

import (
	"fmt"
)

type (
	//matrix key is the column number
	//and each key's value is the value
	//but actually better use the method getVal
	Matrix struct {
		val map[int]*Vector
	}
)

//create a new zero matrix with zeros
func NewZeroMatrix(numRow, numCol int) *Matrix {
	m := map[int]*Vector{}

	//loop from 1 ... numCol to create the key
	for i := 1; i <= numRow; i++ {
		m[i] = NewZeroVector(numCol)
	}

	return &Matrix{
		val: m,
	}
}

//create a new matrix with [][]int as an input
//validate the input first
func validateMatrixInput(input [][]float64) error {
	//the input can't be empty
	if len(input) == 0 {
		return fmt.Errorf("Input matrix should not be empty")
	}

	//get the first element's length
	//this should not be empty
	numCol := len(input[0])
	if numCol == 0 {
		return fmt.Errorf("Empty row vector detected")
	}

	//the other rows length should agree with this
	for i := 1; i < len(input); i++ {
		if numCol != len(input[i]) {
			return fmt.Errorf("Number of columns does not agree")
		}
	}

	return nil
}

func NewMatrix(input [][]float64) (*Matrix, error) {
	//validate the input first
	if err := validateMatrixInput(input); err != nil {
		return nil, err
	}

	var m Matrix
	m.setValue(input)
	return &m, nil
}

//get number of column and row of a matrix
func (m *Matrix) GetColumnNumber() int { return m.val[1].GetLength() }
func (m *Matrix) GetRowNumber() int    { return len(m.val) }

//modify io.Reader print for matrix
func (m *Matrix) String() string {
	sprint := fmt.Sprintf("Num Column: %d\nNum Row: %d\n",
		m.GetColumnNumber(), m.GetRowNumber())

	for row, col := range m.val {
		sprint += fmt.Sprintf("Row %d: %s\n", row, col)
	}

	return sprint
}

//the characteristic of a matrix is that:
//for every column it should have the same row number vice versa
func (m *Matrix) validate() error {
	var numCol, numRow int

	//of course return error if the matrix has no row
	numRow = m.GetRowNumber()
	if numRow == 0 {
		return fmt.Errorf("Matrix is empty")
	}

	//also return error if the length of first value of map is 0
	//or even does not exist
	if _, ok := m.val[1]; !ok {
		return fmt.Errorf("First row element does not exist")
	} else {
		numCol = m.GetColumnNumber()
		if numCol == 0 {
			return fmt.Errorf("First row element is empty")
		}
	}

	//now loop through all the columns to make sure all of them has the same length
	for key, row := range m.val {
		if row.GetLength() != numCol {
			return fmt.Errorf("Number of element for row %d does not match", key)
		}
	}

	return nil
}

//get a single value of a Matrix
//for example to get the 1st column and the 5th row: GetSingleValue(1,5)
func (m *Matrix) getSingleValue(row, col int) float64 {
	return m.val[row].getSingleValue(col)
}

func (m *Matrix) GetSingleValue(row, col int) (float64, error) {
	//proof the existence of the column first
	if _, ok := m.val[row]; !ok {
		return 0, fmt.Errorf("Row %d does not exist", row)
	}

	//and then validate the row slice to avoid panic
	if m.val[row].GetLength() < col {
		return 0, fmt.Errorf("Column %d does not exist", col)
	}

	//get the value from the vector
	//return error if failed

	return m.getSingleValue(row, col), nil
}

//set a single value to a Matrix
//for example to set value 3 to the 1st column and 5th row: SetSingleValue(1,5, 3)
func (m *Matrix) setSingleValue(row, col int, newVal float64) {
	m.val[row].setSingleValue(col, newVal)
}

func (m *Matrix) SetSingleValue(row, col int, newVal float64) error {
	//first get the value for the column and row
	//if error then return error
	if _, err := m.GetSingleValue(row, col); err != nil {
		return err
	}

	m.setSingleValue(row, col, newVal)

	return nil
}

//set value to a matrix
//input is a [][]int which supposed to be converted into map[int]*Vector matrix value
//all of them can't contain any empty slice
func (m *Matrix) setValue(input [][]float64) {
	res := map[int]*Vector{}
	for key, val := range input {
		/*v := NewZeroVector(len(val))
		v.SetValue(val)*/
		res[key+1] = NewVector(val)
	}

	m.val = res
}

//transpose a Matrix and set the result as transposed
//validate only once and then do the transpose
func (m *Matrix) transpose() *Matrix {
	//create new matrix
	//switch length of column and row
	newCol, newRow := m.GetRowNumber(), m.GetColumnNumber()
	t := NewZeroMatrix(newRow, newCol)

	for i := 1; i <= newRow; i++ {
		for j := 1; j <= newCol; j++ {
			//also switch the index from original matrix's value
			fmt.Println(i, j)
			t.setSingleValue(i, j, m.getSingleValue(j, i))
		}
	}

	return t
}

func (m *Matrix) Transpose() (*Matrix, error) {
	if err := m.validate(); err != nil {
		return nil, err
	}

	return m.transpose(), nil
}

//add a variable into a matrix
func (m *Matrix) AddVariable(x float64) {
	for i := 1; i <= m.GetRowNumber(); i++ {
		m.val[i].AddVariable(x)
	}

}

//multiply the matrix with a variable
func (m *Matrix) MultiplyVariable(x float64) {
	for i := 1; i <= m.GetRowNumber(); i++ {
		m.val[i].MultiplyVariable(x)
	}
}

//add two Matrizes
//as validation: dimension of both should agree
//simply add the values from both with the same index
func (m *Matrix) addMatrix(m2 *Matrix) {
	for i := 1; i <= m.GetColumnNumber(); i++ {
		for j := 1; j <= m.GetRowNumber(); j++ {
			newVal := m.getSingleValue(j, i) + m2.getSingleValue(j, i)
			m.setSingleValue(j, i, newVal)
		}
	}
}

func (m *Matrix) AddMatrix(m2 *Matrix) error {
	//make sure both of the dimensions agree
	if m.GetRowNumber() != m2.GetRowNumber() {
		return fmt.Errorf("Row number does not agree")
	}
	if m.GetColumnNumber() != m2.GetColumnNumber() {
		return fmt.Errorf("Column number does not agree")
	}

	m.addMatrix(m2)
	return nil
}
